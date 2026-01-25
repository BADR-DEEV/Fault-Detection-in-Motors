"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import {
  ComposedChart, Line, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid
} from "recharts";
import './globals.css';
import { Activity, Zap, Cpu, RefreshCw, AlertTriangle, CheckCircle, BarChart3, Microscope, Waves } from "lucide-react";

// ==========================================
// 1. PHYSICS ENGINE (Updated for MaFaulDa v2)
// ==========================================
const RAW_FS = 50000; // Must match Python cfg.raw_fs
const DURATION = 1.0; // Seconds (needed for spectral resolution)

const generateHighFidelitySignal = (rpm: number, type: string) => {
  const numSamples = RAW_FS * DURATION;
  const hz = rpm / 60; // Fundamental Frequency
  
  // Arrays for payload
  const tach = new Float32Array(numSamples);
  const ax = new Float32Array(numSamples);
  const ay = new Float32Array(numSamples);
  const az = new Float32Array(numSamples);

  // Random phase offsets for realism
  const phi = Math.random() * 2 * Math.PI;

  for (let i = 0; i < numSamples; i++) {
    const t = i / RAW_FS;
    
    // --- A. TACHOMETER GENERATION (TTL Pulse) ---
    // Python script looks for rising edges. We simulate a 1-pulse-per-rev square wave.
    tach[i] = Math.sin(2 * Math.PI * hz * t) > 0 ? 5.0 : 0.0;

    // --- B. VIBRATION GENERATION ---
    
    // 1. Base Noise (Sensor floor + friction)
    let vx = (Math.random() - 0.5) * 0.02; 
    let vy = (Math.random() - 0.5) * 0.02;
    let vz = (Math.random() - 0.5) * 0.02;

    switch (type) {
      case "normal":
        vx += 0.05 * Math.sin(2 * Math.PI * hz * t + phi);
        vy += 0.05 * Math.sin(2 * Math.PI * hz * t + phi + Math.PI/2);
        vz += 0.02 * Math.sin(2 * Math.PI * hz * t);
        break;

      case "imbalance":
        vx += 0.8 * Math.sin(2 * Math.PI * hz * t + phi);
        vy += 0.6 * Math.sin(2 * Math.PI * hz * t + phi - Math.PI/2); 
        vz += 0.1 * Math.sin(2 * Math.PI * hz * t); 
        break;

      case "misalignment_horizontal":
        vx += 0.4 * Math.sin(2 * Math.PI * hz * t + phi);
        vx += 0.6 * Math.sin(2 * Math.PI * (2 * hz) * t + phi); // 2x Dominant
        vz += 0.3 * Math.sin(2 * Math.PI * hz * t);
        break;

      case "misalignment_vertical":
         vx += 0.5 * Math.sin(2 * Math.PI * hz * t);
         vx += 0.3 * Math.sin(2 * Math.PI * (2 * hz) * t);
         vx += 0.15 * Math.sin(2 * Math.PI * (3 * hz) * t);
         break;
      
      case "bearing_outer":
         const bpfo = hz * 3.58; 
         const impact = Math.pow(Math.max(0, Math.sin(2 * Math.PI * bpfo * t)), 10);
         const resonance = Math.sin(2 * Math.PI * 3000 * t);
         const faultSig = impact * resonance * 0.5;
         vx += faultSig;
         vy += faultSig; 
         vx += 0.05 * Math.sin(2 * Math.PI * hz * t);
         break;
    }

    ax[i] = vx;
    ay[i] = vy;
    az[i] = vz;
  }

  return {
    tach: Array.from(tach),
    ax: Array.from(ax),
    ay: Array.from(ay),
    az: Array.from(az)
  };
};

// ==========================================
// 2. MAIN DASHBOARD
// ==========================================
export default function Dashboard() {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [rpm, setRpm] = useState(1750);
  const [simType, setSimType] = useState("normal");
  const [serverStatus, setServerStatus] = useState(false);
  
  // FIX: Store the generated signal locally so we don't depend on the API echoing it back
  const [localSignal, setLocalSignal] = useState<number[]>([]);

  useEffect(() => {
    const checkServer = async () => {
      try {
        await axios.get('http://127.0.0.1:8000/docs');
        setServerStatus(true);
      } catch (e) { setServerStatus(false); }
    };
    checkServer();
  }, []);

  const triggerSimulation = async () => {
    setLoading(true);

    // 1. Generate Physics-Accurate Data (50kHz)
    const { tach, ax, ay, az } = generateHighFidelitySignal(rpm, simType);
    
    // FIX: Save the raw signal (X-axis) locally for the graph
    setLocalSignal(ax);

    try {
      // 2. Send to Python Backend
      const response = await axios.post("http://127.0.0.1:8000/predict", { 
        tach, 
        ax, 
        ay, 
        az,
        fs: RAW_FS 
      });
      
      setResult(response.data);
      setServerStatus(true);
    } catch (error) {
      console.error(error);
      setServerStatus(false);
      alert("AI Error: Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  // FIX: Safe Data Slicing
  const displaySlice = 2000;
  
  // Logic: Use API returned segment if available, otherwise use local generated signal
  const sourceSignal = (result && result.raw_segment) ? result.raw_segment : localSignal;

  const chartData = sourceSignal.length > 0
    ? sourceSignal.slice(0, displaySlice).map((val: number, idx: number) => ({
        index: idx,
        signal: val,
        // Map attention overlay if available, handling potential length mismatch
        attention: (result && result.xai_saliency && result.xai_saliency[idx]) ? result.xai_saliency[idx] : 0,
      }))
    : [];

  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-200 font-sans selection:bg-cyan-500/30">
        <header className="h-16 border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
          <div className="max-w-7xl mx-auto h-full px-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Cpu className="w-6 h-6 text-cyan-400" />
              <h1 className="text-xl font-bold tracking-wide text-white">
                Vibra<span className="text-cyan-400">Guard</span> <span className="text-xs font-normal text-slate-500 ml-2">v3.0 (Order Domain)</span>
              </h1>
            </div>
            <div className={`flex items-center gap-2 px-4 py-1.5 rounded-full border text-xs font-bold uppercase tracking-wider ${serverStatus ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400" : "border-rose-500/30 bg-rose-500/10 text-rose-400"}`}>
              <Activity className="w-3 h-3" />
              {serverStatus ? "AI Connected" : "AI Disconnected"}
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-12 gap-8">
          {/* CONTROLS */}
          <aside className="col-span-12 lg:col-span-3 space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
              <h2 className="text-xs uppercase tracking-widest text-slate-400 font-bold mb-6 flex items-center gap-2">
                <RefreshCw className="w-4 h-4" /> Digital Twin Control
              </h2>

              <div className="mb-6">
                <label className="block text-xs font-bold text-slate-300 mb-2">MOTOR SPEED (RPM)</label>
                <div className="flex justify-between items-end mb-2">
                  <span className="text-2xl font-mono text-cyan-400 leading-none">{rpm}</span>
                </div>
                <input
                  type="range" min={700} max={3600} step={50}
                  value={rpm} onChange={(e) => setRpm(+e.target.value)}
                  className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
              </div>

              <div className="space-y-2">
                <label className="block text-xs font-bold text-slate-300 mb-2">FAULT SIMULATION</label>
                {[
                  { id: "normal", label: "Healthy State" },
                  { id: "imbalance", label: "Unbalance (1x)" },
                  { id: "misalignment_horizontal", label: "Horiz. Misalign (2x)" },
                  { id: "misalignment_vertical", label: "Vert. Misalign (Comb)" },
                  { id: "bearing_outer", label: "Bearing Outer Race" },
                ].map((item) => (
                  <button
                    key={item.id}
                    onClick={() => setSimType(item.id)}
                    className={`w-full text-left px-4 py-3 rounded-lg text-xs font-bold uppercase transition-all border
                      ${simType === item.id
                        ? "border-cyan-500 bg-cyan-500/10 text-white shadow-[0_0_15px_rgba(6,182,212,0.2)]"
                        : "border-slate-800 bg-slate-800/50 text-slate-400 hover:bg-slate-800"}`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>

              <button
                onClick={triggerSimulation}
                disabled={loading}
                className="w-full mt-6 py-4 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 rounded-xl font-bold text-white flex items-center justify-center gap-2 shadow-lg transition-all active:scale-95 disabled:opacity-50"
              >
                {loading ? <RefreshCw className="animate-spin w-5 h-5" /> : <Zap className="w-5 h-5" />}
                RUN INFERENCE
              </button>
            </div>
            
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                 <h3 className="text-[10px] uppercase font-bold text-slate-500 mb-2">Simulation Stats</h3>
                 <div className="grid grid-cols-2 gap-4 text-xs text-slate-300">
                    <div>
                        <span className="block text-slate-500">Sample Rate</span>
                        {RAW_FS} Hz
                    </div>
                    <div>
                        <span className="block text-slate-500">Duration</span>
                        {DURATION} sec
                    </div>
                    <div>
                        <span className="block text-slate-500">Points Sent</span>
                        {(RAW_FS * DURATION).toLocaleString()}
                    </div>
                 </div>
            </div>
          </aside>

          {/* ANALYTICS */}
          <section className="col-span-12 lg:col-span-9 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <KPICard 
                title="AI PREDICTION" 
                value={result?.prediction?.replace(/_/g, " ") || "--"} 
                color={result?.prediction?.includes("normal") ? "text-emerald-400" : "text-rose-500"}
                icon={result?.prediction?.includes("normal") ? CheckCircle : AlertTriangle}
              />
              <KPICard title="CONFIDENCE" value={result ? `${(result.confidence * 100).toFixed(1)}%` : "--"} color="text-white" icon={BarChart3}/>
              <KPICard title="ESTIMATED RPM (BACKEND)" value={result?.rpm ? Math.round(result.rpm) : "--"} color="text-slate-200" icon={Activity}/>
            </div>

            {/* CHART */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
              <div className="flex justify-between items-end mb-6">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Microscope className="w-5 h-5 text-purple-400"/> Signal Analysis
                  </h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Displaying first {displaySlice/RAW_FS*1000}ms of data. AI processed full 1s buffer.
                  </p>
                </div>
                <div className="flex gap-4 text-[10px] font-bold uppercase text-slate-500">
                    <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-slate-500"></span> Raw Signal (Underhang)</div>
                    <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-rose-500 shadow-[0_0_8px_red]"></span> Salient Feature</div>
                </div>
              </div>

              {localSignal.length > 0 ? (
                <GradCamChart data={chartData} />
              ) : (
                <div className="h-[400px] flex flex-col items-center justify-center text-slate-600 border-2 border-dashed border-slate-800 rounded-xl bg-slate-900/50">
                  <Waves className="w-16 h-16 mb-4 opacity-20" />
                  <p className="text-sm font-medium">System Ready.</p>
                  <p className="text-xs opacity-60">Select a fault condition to generate digital twin telemetry.</p>
                </div>
              )}
            </div>
          </section>
        </main>
      </body>
    </html>
  );
}

// ==========================================
// 3. SUB-COMPONENTS
// ==========================================
const KPICard = ({ title, value, color, icon: Icon }: any) => (
  <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl relative overflow-hidden group">
    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
        <Icon className="w-16 h-16 text-white" />
    </div>
    <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">{title}</p>
    <p className={`text-xl md:text-2xl font-black uppercase ${color} truncate`}>{value}</p>
  </div>
);

// Custom Scatter Dot
const RenderCustomDot = (props: any) => {
  const { cx, cy, payload } = props;
  const importance = payload.attention;

  if (!importance || importance < 0.4) return null; // Only show relevant points

  // Color gradient based on importance
  let fill = "#eab308"; 
  let radius = importance * 2;

  if (importance > 0.7) {
      fill = "#ef4444"; 
      radius = importance * 4;
  }

  return (
      <circle 
          cx={cx} 
          cy={cy} 
          r={radius} 
          fill={fill} 
          opacity={0.8}
          stroke="none"
      />
  );
};

const GradCamChart = ({ data }: { data: any[] }) => {
  return (
    <div className="w-full h-[450px] bg-slate-950/50 rounded-xl border border-slate-800/50 p-2 relative">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" opacity={0.5} vertical={false} />
          <YAxis domain={['auto', 'auto']} tick={{fontSize: 10, fill: '#64748b'}} width={35} tickFormatter={(val) => val.toFixed(1)}/>
          <XAxis dataKey="index" hide />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const val = payload[0].value as number;
                const imp = payload[0].payload.attention as number;
                return (
                  <div className="bg-slate-900 border border-slate-700 p-3 rounded shadow-lg text-xs backdrop-blur-md">
                    <div className="text-slate-300 font-bold mb-1">Index: {payload[0].payload.index}</div>
                    <div className="text-slate-400">Amp: <span className="font-mono text-white">{val.toFixed(3)}</span></div>
                    {imp > 0 && (
                         <div className="text-rose-400 mt-1 font-bold">Attention: {(imp*100).toFixed(0)}%</div>
                    )}
                  </div>
                )
              }
              return null;
            }}
          />
          
          <Line 
            type="monotone" 
            dataKey="signal" 
            stroke="#64748b" 
            strokeWidth={1.5} 
            dot={false}
            animationDuration={500}
          />
          
          <Scatter 
            dataKey="signal" 
            shape={<RenderCustomDot />} 
            animationDuration={500}
            isAnimationActive={false} 
          />
          
        </ComposedChart>
     </ResponsiveContainer>
    </div>
  );
};