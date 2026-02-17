"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import {
  ComposedChart, Line, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, Legend
} from "recharts";
import './globals.css';
import {
  Activity, Zap, Cpu, RefreshCw, AlertTriangle, CheckCircle,
  BarChart3, Microscope, Database, FileSearch, Waves, Gauge, Info
} from "lucide-react";

// CONFIG
const API_URL = "http://127.0.0.1:8000";
const RAW_FS = 50000;

// MAFAULDA FAULT TYPES (matches server exactly)
const FAULT_TYPES = [
  { value: "Normal", label: "✅ Normal" },
  { value: "Imbalance", label: "⚖️ Imbalance" },
  { value: "Horiz_Misalign", label: "↔️ Horizontal Misalignment" },
  { value: "Vert_Misalign", label: "↕️ Vertical Misalignment" },
  { value: "Ball_Fault", label: "⚽ Ball Fault" },
  { value: "Outer_Race", label: "🛞 Outer Race Fault" }
];

export default function Dashboard() {
  const [source, setSource] = useState("simulation");
  const [simType, setSimType] = useState("Normal");
  const [rpm, setRpm] = useState(1750);
  const [loading, setLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState(false);
  const [timeData, setTimeData] = useState<any[]>([]);
  const [specData, setSpecData] = useState<any[]>([]);
  const [result, setResult] = useState<any>(null);
  const [currentFile, setCurrentFile] = useState("");
  const [currentSeverity, setCurrentSeverity] = useState("");
  const [physicsExplanation, setPhysicsExplanation] = useState("");

  useEffect(() => { checkServer(); }, []);

  const checkServer = async () => {
    try { 
      await axios.get(`${API_URL}/health`);
      setServerStatus(true); 
    } catch { 
      setServerStatus(false); 
    }
  };

  const handleRun = async () => {
    setLoading(true);
    try {
      let payload;
      setCurrentSeverity("");

      // 1. GET DATA
      if (source === "dataset") {
        const res = await axios.get(`${API_URL}/get_sample`, { params: { fault_type: simType } });
        payload = res.data;
        setCurrentFile(res.data.filename || "Unknown File");
        setCurrentSeverity(res.data.severity || "");
      } else {
        payload = generatePhysicsAlignedSimulation(rpm, simType);
        setCurrentFile("Synthetic Physics Model");
        setCurrentSeverity(`${simType} @ ${rpm} RPM`);
      }

      // 2. UPDATE TIME CHART (Decimated for UI)
      const decimationFactor = 25;
      const uiData = payload.ax
        .filter((_: any, i: number) => i % decimationFactor === 0)
        .slice(0, 1000)
        .map((v: number, i: number) => ({ 
          time: (i * decimationFactor / RAW_FS * 1000).toFixed(1), // ms
          axial: parseFloat(v.toFixed(4)),
          radial: parseFloat(payload.ay[i * decimationFactor]?.toFixed(4) || 0),
          tangential: parseFloat(payload.az[i * decimationFactor]?.toFixed(4) || 0)
        }));
      setTimeData(uiData);

      // 3. SEND TO AI
      const res = await axios.post(`${API_URL}/predict`, payload);
      setResult(res.data);
      setPhysicsExplanation(res.data.explanation || "Physics analysis unavailable");

      // 4. PROCESS SPECTRUM FOR UI
      if (res.data.spectrum_f && res.data.spectrum_val) {
        const sData = res.data.spectrum_f.map((f: number, i: number) => ({
          freq: parseFloat(f.toFixed(1)),
          amp: parseFloat(res.data.spectrum_val[i].toFixed(4))
        })).filter((d: any) => d.freq <= 2000); // Focus on 0-2kHz bearing range
        setSpecData(sData);
      }

    } catch (e: any) {
      console.error("AI Processing Failed:", e);
      alert(`AI Processing Failed: ${e.response?.data?.detail || e.message}`);
      setResult(null);
      setPhysicsExplanation("");
    } finally {
      setLoading(false);
    }
  };

  return (
    <html>
      <body>
        <div className="min-h-screen bg-slate-950 text-slate-200 font-sans pb-10">
          {/* HEADER */}
          <header className="h-16 border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-6 h-full flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Cpu className="w-6 h-6 text-cyan-400" />
                <h1 className="text-xl font-bold text-white">
                  Vibra<span className="text-cyan-400">Guard</span> <span className="text-xs text-slate-500 font-normal">MaFaulDa Physics Edition</span>
                </h1>
              </div>
              <div className={`px-3 py-1 rounded-full border text-xs font-bold uppercase ${serverStatus ? "border-emerald-500/30 text-emerald-400" : "border-rose-500/30 text-rose-400"}`}>
                {serverStatus ? "✅ AI SYSTEM READY (MaFaulDa Physics-Validated)" : "❌ BACKEND OFFLINE"}
              </div>
            </div>
          </header>

          <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-12 gap-8">
            {/* SIDEBAR */}
            <aside className="col-span-12 lg:col-span-3 space-y-6">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                <h2 className="text-xs font-bold text-slate-400 mb-4 flex gap-2"><RefreshCw className="w-4 h-4" /> INPUT SOURCE</h2>

                <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800 mb-6">
                  {['simulation', 'dataset'].map(m => (
                    <button 
                      key={m} 
                      onClick={() => { setSource(m); setResult(null); setPhysicsExplanation(""); }}
                      className={`flex-1 py-2 text-xs font-bold uppercase rounded ${source === m ? "bg-cyan-600 text-white" : "text-slate-500"}`}
                    >
                      {m.toUpperCase()}
                    </button>
                  ))}
                </div>

                <div className="space-y-3 mb-6">
                  <label className="text-xs font-bold text-slate-500 flex items-center gap-2">
                    <Database className="w-3 h-3" /> FAULT CONDITION
                  </label>
                  
                  {source === 'dataset' ? (
                    <select
                      value={simType}
                      onChange={(e) => { setSimType(e.target.value); setResult(null); setPhysicsExplanation(""); }}
                      className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-white text-xs font-bold"
                    >
                      {FAULT_TYPES.map(t => (
                        <option key={t.value} value={t.value}>
                          {t.label}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <div className="grid grid-cols-2 gap-2">
                      {FAULT_TYPES.slice(0, 4).map(t => (
                        <button 
                          key={t.value} 
                          onClick={() => { setSimType(t.value); setResult(null); setPhysicsExplanation(""); }}
                          className={`text-left px-3 py-2 rounded border text-[10px] font-bold uppercase
                            ${simType === t.value ? "border-cyan-500 bg-cyan-500/10 text-white" : "border-slate-800 text-slate-400 hover:bg-slate-800/50"}`}
                        >
                          {t.label.split(' ')[0]}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {source === 'simulation' && (
                  <div className="mb-6">
                    <label className="text-xs font-bold text-slate-500 flex justify-between">
                      <span className="flex items-center gap-1"><Gauge className="w-3 h-3" /> RPM</span>
                      <span>{rpm} RPM</span>
                    </label>
                    <input 
                      type="range" 
                      min={600} 
                      max={3000} 
                      value={rpm} 
                      onChange={e => { setRpm(+e.target.value); setResult(null); setPhysicsExplanation(""); }}
                      className="w-full accent-cyan-500" 
                    />
                    <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                      <span>600</span>
                      <span>1750 (Typical)</span>
                      <span>3000</span>
                    </div>
                  </div>
                )}

                <button 
                  onClick={handleRun} 
                  disabled={loading || !serverStatus}
                  className="w-full py-4 bg-cyan-600 hover:bg-cyan-500 rounded-xl font-bold text-white shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  {loading ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      PROCESSING VIBRATION DATA...
                    </span>
                  ) : (
                    <span className="flex items-center justify-center gap-2">
                      <Zap className="w-5 h-5" /> ANALYZE VIBRATION
                    </span>
                  )}
                </button>
                
                <div className="mt-4 p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                  <p className="text-[10px] text-slate-400 flex items-start gap-2">
                    <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
                    <span>
                      Physics-validated on MaFaulDa dataset • Radial-dominant misalignment detection • Severity-stratified analysis
                    </span>
                  </p>
                </div>
              </div>
            </aside>

            {/* MAIN CONTENT */}
            <section className="col-span-12 lg:col-span-9 space-y-6">
              {/* KPI CARDS */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <KPICard 
                  title="PREDICTION" 
                  value={result?.prediction || "--"}
                  color={result?.prediction === "Normal" ? "text-emerald-400" : "text-rose-500"}
                  icon={result?.prediction === "Normal" ? CheckCircle : AlertTriangle} 
                />
                <KPICard 
                  title="CONFIDENCE" 
                  value={result ? `${(result.confidence * 100).toFixed(1)}%` : "--"} 
                  color="text-white" 
                  icon={BarChart3} 
                />
                <KPICard 
                  title="RPM DETECTED" 
                  value={result?.rpm ? Math.round(result.rpm) : "--"} 
                  color="text-blue-300" 
                  icon={Gauge} 
                />
                <KPICard 
                  title="SEVERITY" 
                  value={currentSeverity || "--"} 
                  color="text-amber-400" 
                  icon={Activity} 
                />
              </div>

              {/* PHYSICS EXPLANATION CARD */}
              {physicsExplanation && (
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                  <div className="flex items-start gap-3 mb-3">
                    <Microscope className="w-5 h-5 text-cyan-400 mt-0.5 flex-shrink-0" />
                    <h3 className="text-sm font-bold">PHYSICS-BASED DIAGNOSIS EXPLANATION</h3>
                  </div>
                  <p className="text-sm text-slate-300 leading-relaxed bg-slate-800/30 p-4 rounded-lg border border-slate-700">
                    {physicsExplanation}
                  </p>
                  <div className="mt-3 text-[10px] text-slate-500 italic">
                    Explanation generated using MaFaulDa physics validation results (Axis Ablation Test, Severity Stratification)
                  </div>
                </div>
              )}

              {/* TIME DOMAIN CHART */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                <div className="flex justify-between mb-4">
                  <h3 className="text-sm font-bold flex gap-2">
                    <Waves className="w-4 h-4 text-blue-400" /> TIME WAVEFORM (Multi-Axis)
                  </h3>
                  <span className="text-xs text-slate-500">
                    {currentFile} {currentSeverity && `• Severity: ${currentSeverity}`}
                  </span>
                </div>
                <div className="h-60">
                  <ResponsiveContainer>
                    <ComposedChart data={timeData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                      <XAxis 
                        dataKey="time" 
                        tick={{ fontSize: 10 }} 
                        label={{ value: 'Time (ms)', position: 'insideBottom', fontSize: 10, fill: '#64748b' }} 
                        dy={10}
                      />
                      <YAxis 
                        tick={{ fontSize: 10 }} 
                        label={{ value: 'Amplitude (g)', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#64748b' }} 
                        dx={-10}
                        domain={['auto', 'auto']}
                      />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} 
                        labelStyle={{ color: '#cbd5e1' }}
                        itemStyle={{ color: '#fff' }}
                      />
                      <Legend 
                        wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} 
                        formatter={(value) => <span className="text-slate-300">{value}</span>}
                      />
                      <Line type="monotone" dataKey="axial" stroke="#f87171" strokeWidth={1.8} dot={false} name="Axial" />
                      <Line type="monotone" dataKey="radial" stroke="#38bdf8" strokeWidth={1.8} dot={false} name="Radial" />
                      <Line type="monotone" dataKey="tangential" stroke="#a78bfa" strokeWidth={1.8} dot={false} name="Tangential" />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-3 flex gap-4 text-[10px] text-slate-500">
                  <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-rose-500"></div> Axial</span>
                  <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-cyan-400"></div> Radial</span>
                  <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-purple-400"></div> Tangential</span>
                </div>
              </div>

              {/* FREQUENCY DOMAIN CHART */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                <div className="flex justify-between mb-4">
                  <h3 className="text-sm font-bold flex gap-2">
                    <Microscope className="w-4 h-4 text-purple-400" /> FREQUENCY SPECTRUM (Physics-Aligned)
                  </h3>
                  <div className="flex gap-4 text-[10px] text-slate-500">
                    <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-purple-500"></div> PSD</span>
                    <span className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-cyan-400 border border-dashed"></div> Harmonics</span>
                  </div>
                </div>
                <div className="h-72">
                  {specData.length > 0 ? (
                    <ResponsiveContainer>
                      <ComposedChart data={specData} margin={{ top: 10, right: 30, bottom: 10, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} vertical={false} />
                        <XAxis 
                          dataKey="freq" 
                          tick={{ fontSize: 10 }} 
                          label={{ value: 'Frequency (Hz)', position: 'insideBottomRight', fontSize: 10, fill: '#64748b' }} 
                          dx={15}
                        />
                        <YAxis 
                          tick={{ fontSize: 10 }} 
                          label={{ value: 'PSD (dB/Hz)', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#64748b' }} 
                          dx={-10}
                        />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} 
                          labelFormatter={(label) => `Frequency: ${label} Hz`}
                          itemStyle={{ color: '#fff' }}
                        />
                        <Area 
                          type="monotone" 
                          dataKey="amp" 
                          fill="#8b5cf6" 
                          stroke="#7c3aed" 
                          fillOpacity={0.3} 
                          name="Power Spectral Density"
                        />
                        
                        {/* Physics-Based Harmonic Markers */}
                        {result?.rpm && renderHarmonicMarkers(result.prediction, result.rpm)}
                      </ComposedChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-slate-600 text-sm border border-dashed border-slate-800 rounded bg-slate-800/20">
                      <FileSearch className="w-12 h-12 mb-3 opacity-50" />
                      <p className="font-medium">WAITING FOR VIBRATION ANALYSIS</p>
                      <p className="text-xs mt-1">Run analysis to see physics-aligned spectrum with harmonic markers</p>
                    </div>
                  )}
                </div>
                <div className="mt-3 text-[10px] text-slate-500 italic">
                  Spectrum shows 0-2000 Hz range (critical for bearing fault detection). Harmonic markers adapt to detected fault type.
                </div>
              </div>
            </section>
          </main>
          
          <footer className="max-w-7xl mx-auto px-6 py-6 text-center border-t border-slate-800 text-[10px] text-slate-500">
            VibraGuard MaFaulDa Edition • Physics-Validated AI for Motor Fault Diagnosis • 
            Based on MaFaulDa Dataset (Section 3.2: Radial-Dominant Misalignment Physics)
          </footer>
        </div>
      </body>
    </html>
  );
}

// PHYSICS-ALIGNED HARMONIC MARKERS (MaFaulDa Specific)
const renderHarmonicMarkers = (faultType: string, rpm: number) => {
  const fundamental = rpm / 60;
  const BPFO = 2.998 * fundamental; // Outer race coefficient
  const BSF = 1.871 * fundamental;   // Ball spin coefficient
  
  const markers: any[] = [];
  
  switch(faultType) {
    case "Imbalance":
      // 1x RPM harmonic dominant
      [1, 2, 3].forEach(m => {
        const freq = m * fundamental;
        if (freq <= 2000) {
          markers.push(
            <ReferenceLine 
              key={`imb-${m}`} 
              x={freq} 
              stroke="#f87171" 
              strokeDasharray="3 3" 
              label={{ 
                value: `${m}x RPM`, 
                position: 'top', 
                fill: '#f87171', 
                fontSize: 10,
                offset: 5
              }} 
            />
          );
        }
      });
      break;
      
    case "Horiz_Misalign":
    case "Vert_Misalign":
      // 2x, 3x RPM harmonics dominant (misalignment signature)
      [2, 3, 4].forEach(m => {
        const freq = m * fundamental;
        if (freq <= 2000) {
          markers.push(
            <ReferenceLine 
              key={`mis-${m}`} 
              x={freq} 
              stroke="#fbbf24" 
              strokeDasharray="4 2" 
              label={{ 
                value: `${m}x RPM`, 
                position: 'top', 
                fill: '#fbbf24', 
                fontSize: 10,
                offset: 5
              }} 
            />
          );
        }
      });
      break;
      
    case "Ball_Fault":
      // BSF harmonics (2x-4x)
      [2, 3, 4].forEach(m => {
        const freq = m * BSF;
        if (freq <= 2000) {
          markers.push(
            <ReferenceLine 
              key={`bsf-${m}`} 
              x={freq} 
              stroke="#6ee7b7" 
              strokeDasharray="2 4" 
              label={{ 
                value: `${m}x BSF`, 
                position: 'top', 
                fill: '#6ee7b7', 
                fontSize: 9,
                offset: 5
              }} 
            />
          );
        }
      });
      break;
      
    case "Outer_Race":
      // BPFO harmonics (2x-4x)
      [2, 3, 4].forEach(m => {
        const freq = m * BPFO;
        if (freq <= 2000) {
          markers.push(
            <ReferenceLine 
              key={`bpfo-${m}`} 
              x={freq} 
              stroke="#93c5fd" 
              strokeDasharray="5 5" 
              label={{ 
                value: `${m}x BPFO`, 
                position: 'top', 
                fill: '#93c5fd', 
                fontSize: 9,
                offset: 5
              }} 
            />
          );
        }
      });
      break;
      
    default:
      // Show fundamental harmonics as reference
      [1, 2, 3].forEach(m => {
        const freq = m * fundamental;
        if (freq <= 2000) {
          markers.push(
            <ReferenceLine 
              key={`ref-${m}`} 
              x={freq} 
              stroke="#94a3b8" 
              strokeDasharray="3 3" 
              opacity={0.6}
            />
          );
        }
      });
  }
  
  return markers;
};

// PHYSICS-ALIGNED SIMULATION (Matches MaFaulDa Physics Validation)
const generatePhysicsAlignedSimulation = (rpm: number, type: string) => {
  const samples = 50000;
  const hz = rpm / 60;
  const t = Array.from({ length: samples }, (_, i) => i / RAW_FS);
  
  // Initialize axis arrays
  const ax = new Array(samples).fill(0);
  const ay = new Array(samples).fill(0);
  const az = new Array(samples).fill(0);
  
  // Base noise (realistic sensor noise)
  const noise = () => (Math.random() - 0.5) * 0.015;
  
  // Generate physics-aligned signals per fault type
  for (let i = 0; i < samples; i++) {
    const time = t[i];
    
    // Add base noise to all axes
    ax[i] = noise();
    ay[i] = noise();
    az[i] = noise();
    
    switch(type) {
      case 'Normal':
        // Low-amplitude balanced vibration
        ax[i] += 0.02 * Math.sin(2 * Math.PI * hz * time);
        ay[i] += 0.02 * Math.sin(2 * Math.PI * hz * time + 0.3);
        az[i] += 0.015 * Math.sin(2 * Math.PI * hz * time - 0.2);
        break;
        
      case 'Imbalance':
        // RADIAL-DOMINANT at severe stages (MaFaulDa physics validation)
        // Mild: multi-axis | Severe: radial dominance
        const radialStrength = rpm > 2000 ? 0.6 : 0.3; // Severity-dependent
        ay[i] += radialStrength * Math.sin(2 * Math.PI * hz * time); // Strong radial (1x RPM)
        ax[i] += 0.1 * Math.sin(2 * Math.PI * hz * time); // Weaker axial
        az[i] += 0.08 * Math.sin(2 * Math.PI * hz * time); // Weaker tangential
        break;
        
      case 'Horiz_Misalign':
        // RADIAL-DOMINANT due to coupling dynamics (MaFaulDa Section 3.2)
        // 2x RPM harmonic dominant in radial direction
        ay[i] += 0.4 * Math.sin(2 * Math.PI * 2 * hz * time); // Strong radial 2x
        ax[i] += 0.25 * Math.sin(2 * Math.PI * 2 * hz * time); // Moderate axial 2x
        break;
        
      case 'Vert_Misalign':
        // Axial component stronger but radial still significant
        ax[i] += 0.35 * Math.sin(2 * Math.PI * 2 * hz * time);
        ay[i] += 0.2 * Math.sin(2 * Math.PI * 2 * hz * time);
        break;
        
      case 'Ball_Fault':
        // Impulsive signatures at BSF harmonics (high frequency)
        const bsf = 1.871 * hz; // Ball Spin Frequency
        const impactEnvelope = Math.max(0, 0.25 * (1 + Math.sin(2 * Math.PI * bsf * time * 5)));
        const highFreq = 800 + Math.random() * 200; // Bearing resonance range
        ax[i] += impactEnvelope * Math.sin(2 * Math.PI * highFreq * time) * 0.3;
        ay[i] += impactEnvelope * Math.sin(2 * Math.PI * highFreq * time) * 0.25;
        az[i] += impactEnvelope * Math.sin(2 * Math.PI * highFreq * time) * 0.2;
        break;
        
      case 'Outer_Race':
        // Impulsive signatures at BPFO harmonics
        const bpfo = 2.998 * hz; // Ball Pass Frequency Outer
        const bpfoEnvelope = Math.max(0, 0.2 * (1 + Math.sin(2 * Math.PI * bpfo * time * 4)));
        ax[i] += bpfoEnvelope * Math.sin(2 * Math.PI * 600 * time) * 0.25;
        ay[i] += bpfoEnvelope * Math.sin(2 * Math.PI * 600 * time) * 0.2;
        break;
    }
  }
  
  // Generate realistic tachometer signal (1 pulse/revolution)
  const tach = t.map(time => {
    const phase = (2 * Math.PI * hz * time) % (2 * Math.PI);
    return (phase < 0.2) ? 5.0 : 0.1 + Math.random() * 0.05; // Clean pulse with noise
  });
  
  return { tach, ax, ay, az, fs: RAW_FS };
};

const KPICard = ({ title, value, color, icon: Icon }: any) => (
  <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl relative overflow-hidden">
    <div className="absolute top-3 right-3 opacity-10">
      <Icon className="w-10 h-10" />
    </div>
    <p className="text-[10px] font-bold text-slate-500 mb-1 uppercase tracking-wider">{title}</p>
    <p className={`text-2xl font-black ${color} min-h-[28px]`}>{value}</p>
  </div>
);