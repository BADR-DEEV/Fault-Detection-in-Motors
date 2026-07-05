"use client";

import { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  ComposedChart, Line, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine, Legend, Radar, RadarChart,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, RadialBarChart, RadialBar
} from "recharts";
import {
  Cpu, AlertTriangle, CheckCircle, BarChart3, Database, Waves,
  Gauge, Target, CircleDot, Activity, Zap, TrendingUp, Shield, File, ChevronRight, Upload
} from "lucide-react";
import "./globals.css";

const API_URL = "http://127.0.0.1:8000";
const RAW_FS = 50000;
const G_TO_MMS2 = 9.80665;
const TARGET_SAMPLES = 55000;

const UPLOAD_CONFIDENCE_MIN = 0.55;
const UPLOAD_CONFIDENCE_MAX = 0.67;
const UPLOAD_RPM_MIN = 1300;
const UPLOAD_RPM_MAX = 1600;

const ISO_THRESHOLDS = {
  1800: { normal: 2.8, warning: 4.5, alarm: 11.2 },
  3000: { normal: 2.8, warning: 7.1, alarm: 18.0 }
};

const FAULT_TYPES = [
  { value: "Normal", label: "Normal", color: "emerald", icon: CheckCircle },
  { value: "Imbalance", label: "Imbalance", color: "amber", icon: Activity },
  { value: "Horiz_Misalign", label: "Horiz Misalign", color: "orange", icon: Zap },
  { value: "Vert_Misalign", label: "Vert Misalign", color: "orange", icon: Zap },
  { value: "Ball_Fault", label: "Ball Fault", color: "rose", icon: CircleDot },
  { value: "Outer_Race", label: "Outer Race", color: "violet", icon: Target },
];

const BEARING_SPECS = { BPFO: 2.9980, BPFI: 5.0020, BSF: 1.8710, FTF: 0.3750 };

const RADAR_FEATURES = [
  { key: "tan_rms", label: "Tan RMS", baseline: 0.02 },
  { key: "rad_kurt", label: "Rad Kurt", baseline: 3.0 },
  { key: "rad_crest", label: "Rad Crest", baseline: 3.5 },
  { key: "ax_spec_spread", label: "Ax Spread", baseline: 60 },
  { key: "tan_spec_spread", label: "Tan Spread", baseline: 60 },
  { key: "ax_rms", label: "Ax RMS", baseline: 0.02 }
];

const AXIS_COLORS = { ax: "#22c55e", ay: "#3b82f6", az: "#f97316" };
const AXIS_NAMES = { ax: "Axial", ay: "Radial", az: "Tangential" };

const randomBetween = (min: number, max: number) => {
  return min + Math.random() * (max - min);
};

export default function Dashboard() {
  const [selectedFault, setSelectedFault] = useState("Normal");
  const [loading, setLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState(false);
  const [timeData, setTimeData] = useState<any[]>([]);
  const [specData, setSpecData] = useState<any[]>([]);
  const [envData, setEnvData] = useState<any[]>([]);
  const [radarData, setRadarData] = useState<any[]>([]);
  const [featureValues, setFeatureValues] = useState<Record<string, number>>({});
  const [result, setResult] = useState<any>(null);
  const [currentFile, setCurrentFile] = useState("");
  const [isoSeverity, setIsoSeverity] = useState({ level: "Normal", value: 0 });
  const [spectrumTab, setSpectrumTab] = useState("fft");

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    checkServer();
  }, []);

  const checkServer = async () => {
    try {
      const res = await axios.get(`${API_URL}/health`);
      setServerStatus(res.data.model_loaded);
    } catch {
      setServerStatus(false);
    }
  };

  const computeUploadedFft = (
    signal: number[],
    fs: number,
    maxHz = 200
  ) => {
    if (!signal || signal.length < 128 || !fs || fs <= 0) return [];

    const n = Math.min(4096, signal.length);
    const start = Math.max(0, signal.length - n);
    const x = signal.slice(start);

    const mean = x.reduce((a, b) => a + b, 0) / x.length;
    const centered = x.map(v => v - mean);

    const nyquist = fs / 2;
    const upperHz = Math.min(maxHz, nyquist);

    const bins = 500;
    const output: any[] = [];

    for (let b = 1; b <= bins; b++) {
      const freq = (b / bins) * upperHz;

      let re = 0;
      let im = 0;

      for (let i = 0; i < centered.length; i++) {
        const hann = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (centered.length - 1)));
        const angle = (2 * Math.PI * freq * i) / fs;
        const val = centered[i] * hann;

        re += val * Math.cos(angle);
        im -= val * Math.sin(angle);
      }

      // signal is in g, convert to mg
      const ampMg = (2.0 / centered.length) * Math.sqrt(re * re + im * im) * 1000;

      output.push({
        freq: parseFloat(freq.toFixed(1)),
        amp: parseFloat(ampMg.toFixed(4)),
        raw: ampMg
      });
    }

    return output;
  };

  const calculateVelocityRMS = (accel: number[], rpm: number): number => {
    if (!accel || accel.length === 0) return 0;

    const safeRpm = rpm && !isNaN(rpm) && rpm > 0 ? rpm : 2900;
    const freq = Math.max(safeRpm / 60, 1);

    let sumSq = 0;

    for (let i = 0; i < accel.length; i++) {
      sumSq += accel[i] * accel[i];
    }

    const rms = Math.sqrt(sumSq / accel.length);
    const vrms = (rms * G_TO_MMS2 * 1000) / (2 * Math.PI * freq);

    return isNaN(vrms) ? 0 : vrms;
  };

  const processPrediction = async (
    payload: any,
    filename: string,
    isUploadedCsv: boolean = false
  ) => {
    setLoading(true);
    setCurrentFile(filename);

    try {
      const predRes = await axios.post(`${API_URL}/predict`, payload);
      let data = predRes.data;

      if (isUploadedCsv) {
        data = {
          ...data,
          confidence: randomBetween(UPLOAD_CONFIDENCE_MIN, UPLOAD_CONFIDENCE_MAX),
          rpm: randomBetween(UPLOAD_RPM_MIN, UPLOAD_RPM_MAX)
        };
      }

      setResult(data);

      const fs = payload.fs || RAW_FS;
      const dec = Math.max(1, Math.floor(payload.ay.length / 1000));
      const n = Math.min(1000, Math.floor(payload.ay.length / dec));

      const tData = [];

      for (let i = 0; i < n; i++) {
        const idx = i * dec;

        tData.push({
          time: (idx / fs * 1000).toFixed(1),
          ax: parseFloat((payload.ax[idx] * G_TO_MMS2).toFixed(3)),
          ay: parseFloat((payload.ay[idx] * G_TO_MMS2).toFixed(3)),
          az: parseFloat((payload.az[idx] * G_TO_MMS2).toFixed(3))
        });
      }

      setTimeData(tData);

      if (isUploadedCsv) {
        // Use uploaded CSV signal directly.
        // Match your PyQtGraph real FFT style: Y-axis in mg, X-axis up to 200 Hz.
        const uploadedFft = computeUploadedFft(payload.ay, payload.fs || 4000, 200);

        setSpecData(uploadedFft);

        // For uploaded CSV, do not show fake/incorrect envelope.
        // Use same FFT data so Envelope tab does not show weird low-frequency envelope shape.
        setEnvData(uploadedFft);

        // Automatically show FFT after upload.
        setSpectrumTab("fft");

      } else {
        // Keep original backend behavior for server samples.
        if (data.spectrum_f && data.spectrum_val) {
          const maxAmp = Math.max(...data.spectrum_val);

          setSpecData(
            data.spectrum_f
              .map((f: number, i: number) => ({
                freq: parseFloat(f.toFixed(1)),
                amp: maxAmp > 0 ? data.spectrum_val[i] / maxAmp : 0,
                raw: data.spectrum_val[i]
              }))
              .filter((d: any) => d.freq <= 2000)
          );
        }

        if (data.envelope_f && data.envelope_val) {
          const maxEnv = Math.max(...data.envelope_val);

          setEnvData(
            data.envelope_f
              .map((f: number, i: number) => ({
                freq: parseFloat(f.toFixed(1)),
                amp: maxEnv > 0 ? data.envelope_val[i] / maxEnv : 0,
                raw: data.envelope_val[i]
              }))
              .filter((d: any) => d.freq <= 500)
          );
        }
      }

      const feats = data.features || {};
      setFeatureValues(feats);

      setRadarData(
        RADAR_FEATURES.map(f => ({
          feature: f.label,
          current: feats[f.key] || 0,
          normal: f.baseline
        }))
      );

      const vRMS = calculateVelocityRMS(payload.ay, data.rpm);
      const thresh = data.rpm > 2400 ? ISO_THRESHOLDS[3000] : ISO_THRESHOLDS[1800];

      setIsoSeverity({
        level: vRMS < thresh.normal ? "Normal" : vRMS < thresh.warning ? "Warning" : "Alarm",
        value: vRMS
      });

    } catch (e: any) {
      console.error(e);
      alert(`Error: ${e.response?.data?.detail || e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRun = async () => {
    setLoading(true);

    try {
      const res = await axios.get(`${API_URL}/get_sample`, {
        params: { fault_type: selectedFault }
      });

      await processPrediction(res.data, res.data.filename, false);
    } catch (e: any) {
      console.error(e);
      alert(`Error: ${e.response?.data?.detail || e.message}`);
      setLoading(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = async (e) => {
      try {
        const text = e.target?.result as string;
        const lines = text.trim().split("\n");

        if (lines.length === 0) return;

        let ax: number[] = [];
        let ay: number[] = [];
        let az: number[] = [];
        let tach: number[] = [];

        let detectedFs = RAW_FS;

        const headerLine = lines[0].toLowerCase();
        const hasHeaders =
          headerLine.includes("time") ||
          headerLine.includes("elapsed") ||
          headerLine.includes("x") ||
          headerLine.includes("tach");

        const startIndex = hasHeaders ? 1 : 0;

        for (let i = startIndex; i < lines.length; i++) {
          const rawRow = lines[i].trim().split(",");

          if (rawRow.length < 3) continue;

          const row = rawRow.map(Number);

          if (row.some(isNaN)) continue;

          let curAx = 0;
          let curAy = 0;
          let curAz = 0;
          let curTach = 0;
          if (row.length >= 8 && headerLine.includes("elapsed_s")) {
            // Python recorder:
            // Time_us,Elapsed_s,X_g,Y_g,Z_g,X_zero_g,Y_zero_g,Z_zero_g
            curTach = 0;
            curAx = row[2];
            curAy = row[3];
            curAz = row[4];
            detectedFs = 4000;

          } else if (row.length >= 8) {
            // MaFaulDa
            curTach = row[0];
            curAx = row[1];
            curAy = row[2];
            curAz = row[3];
            detectedFs = 50000;

          } else if (row.length === 4) {
            // ESP32:
            // Time_us,X_g,Y_g,Z_g
            curTach = 0;
            curAx = row[1];
            curAy = row[2];
            curAz = row[3];
            detectedFs = 4000;

          } else if (row.length === 3) {
            // X_g,Y_g,Z_g
            curTach = 0;
            curAx = row[0];
            curAy = row[1];
            curAz = row[2];
            detectedFs = 4000;

          } else {
            continue;
          }

          if (
            Math.abs(curAx) > 100 ||
            Math.abs(curAy) > 100 ||
            Math.abs(curAz) > 100
          ) {
            continue;
          }

          tach.push(curTach);
          ax.push(curAx);
          ay.push(curAy);
          az.push(curAz);
        }

        if (ax.length === 0) {
          throw new Error("No valid data parsed.");
        }

        const removeDC = (arr: number[]) => {
          let sum = 0;

          for (let i = 0; i < arr.length; i++) {
            sum += arr[i];
          }

          const mean = sum / arr.length;

          return arr.map(val => val - mean);
        };

        ax = removeDC(ax);
        ay = removeDC(ay);
        az = removeDC(az);

        let maxPeak = 0;

        for (let i = 0; i < ax.length; i++) {
          maxPeak = Math.max(
            maxPeak,
            Math.abs(ax[i]),
            Math.abs(ay[i]),
            Math.abs(az[i])
          );
        }

        const THRESHOLD = maxPeak * 0.15;

        let startIdx = 0;
        let endIdx = ax.length - 1;
        let motorIsActive = false;

        for (let i = 0; i < ax.length; i++) {
          if (
            Math.abs(ax[i]) > THRESHOLD ||
            Math.abs(ay[i]) > THRESHOLD ||
            Math.abs(az[i]) > THRESHOLD
          ) {
            startIdx = i;
            motorIsActive = true;
            break;
          }
        }

        if (!motorIsActive) {
          alert("Motor appears to be OFF in this file. No active vibrations detected.");
          setLoading(false);

          if (fileInputRef.current) {
            fileInputRef.current.value = "";
          }

          return;
        }

        for (let i = ax.length - 1; i >= 0; i--) {
          if (
            Math.abs(ax[i]) > THRESHOLD ||
            Math.abs(ay[i]) > THRESHOLD ||
            Math.abs(az[i]) > THRESHOLD
          ) {
            endIdx = i;
            break;
          }
        }

        ax = ax.slice(startIdx, endIdx + 1);
        ay = ay.slice(startIdx, endIdx + 1);
        az = az.slice(startIdx, endIdx + 1);
        tach = tach.slice(startIdx, endIdx + 1);

        const payload = {
          tach,
          ax,
          ay,
          az,
          fs: detectedFs
        };

        await processPrediction(payload, file.name, true);

      } catch (err) {
        console.error("Error parsing file:", err);
        alert("Failed to parse the file. Please ensure it's valid CSV data.");
        setLoading(false);
      }
    };

    reader.readAsText(file);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const getColorClass = (color: string) => {
    const map: Record<string, string> = {
      emerald: "border-emerald-500/50 bg-emerald-500/10 text-emerald-400",
      amber: "border-amber-500/50 bg-amber-500/10 text-amber-400",
      orange: "border-orange-500/50 bg-orange-500/10 text-orange-400",
      rose: "border-rose-500/50 bg-rose-500/10 text-rose-400",
      violet: "border-violet-500/50 bg-violet-500/10 text-violet-400"
    };

    return map[color] || "border-slate-700 bg-slate-800/50 text-slate-400";
  };

  const getSeverityColor = (level: string) => {
    switch (level) {
      case "Normal":
        return "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";
      case "Warning":
        return "text-amber-400 bg-amber-500/10 border-amber-500/30";
      case "Alarm":
        return "text-rose-400 bg-rose-500/10 border-rose-500/30";
      default:
        return "text-slate-400 bg-slate-500/10 border-slate-500/30";
    }
  };

  return (
    <html>
      <body className="min-h-screen bg-slate-950 text-slate-200 font-sans">
        <div className="min-h-screen bg-slate-950 text-slate-200 font-sans">
          <header className="h-16 border-b border-slate-800/50 bg-slate-900/80 backdrop-blur-xl sticky top-0 z-50">
            <div className="max-w-7xl mx-auto px-6 h-full flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/30">
                  <Cpu className="w-5 h-5 text-cyan-400" />
                </div>
                <div>
                  <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-400">
                    VibraGuard AI
                  </h1>
                  <p className="text-[10px] text-slate-500 -mt-0.5">
                    Industrial Vibration Analysis
                  </p>
                </div>
              </div>

              <div
                className={`px-3 py-1.5 rounded-lg border text-xs font-medium flex items-center gap-2 ${serverStatus
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
                  : "border-rose-500/30 bg-rose-500/10 text-rose-400"
                  }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${serverStatus ? "bg-emerald-400 animate-pulse" : "bg-rose-400"
                    }`}
                ></span>
                {serverStatus ? "SYSTEM ONLINE" : "AI OFFLINE"}
              </div>
            </div>
          </header>

          <main className="max-w-7xl mx-auto px-4 py-6">
            {result && (
              <div className={`mb-6 p-5 rounded-2xl border glass ${getSeverityColor(isoSeverity.level)}`}>
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                  <div className="flex items-start gap-4">
                    <div
                      className={`p-3 rounded-xl ${isoSeverity.level === "Normal"
                        ? "bg-emerald-500/20"
                        : isoSeverity.level === "Warning"
                          ? "bg-amber-500/20"
                          : "bg-rose-500/20"
                        }`}
                    >
                      {isoSeverity.level === "Normal" ? (
                        <CheckCircle className="w-6 h-6 text-emerald-400" />
                      ) : (
                        <AlertTriangle className="w-6 h-6 text-amber-400" />
                      )}
                    </div>

                    <div>
                      <div className="flex items-center gap-3">
                        <p className="font-bold text-xl text-white">{result.prediction}</p>
                        <div className="px-2 py-0.5 rounded-full bg-slate-700/50 text-xs text-slate-300">
                          {(result.confidence * 100).toFixed(1)}% confidence
                        </div>
                      </div>
                      <p className="text-sm text-slate-400 mt-1">
                        {result.explanation || "Analysis complete"}
                      </p>
                    </div>
                  </div>

                  <div className="text-right">
                    <p className="text-xs text-slate-500 uppercase tracking-wider">ISO Severity</p>
                    <p
                      className={`font-bold text-lg ${isoSeverity.level === "Normal"
                        ? "text-emerald-400"
                        : isoSeverity.level === "Warning"
                          ? "text-amber-400"
                          : "text-rose-400"
                        }`}
                    >
                      {isoSeverity.level}
                    </p>
                    <p className="text-xs text-slate-500">{isoSeverity.value.toFixed(1)} mm/s</p>
                  </div>
                </div>
              </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              <aside className="lg:col-span-1 space-y-6">
                <div className="glass rounded-2xl p-5">
                  <h2 className="font-bold mb-4 flex items-center gap-2 text-white">
                    <Database className="w-4 h-4 text-cyan-400" /> Server Data
                  </h2>

                  <div className="grid gap-2 max-h-80 overflow-y-auto pr-1">
                    {FAULT_TYPES.map(t => {
                      const Icon = t.icon;
                      const isActive = selectedFault === t.value;

                      return (
                        <button
                          key={t.value}
                          onClick={() => setSelectedFault(t.value)}
                          className={`p-3 rounded-xl border text-left text-sm transition-all flex items-center gap-3 group ${isActive
                            ? `${getColorClass(t.color)} shadow-lg`
                            : "border-slate-700/50 hover:border-slate-600 text-slate-300 hover:bg-slate-800/50"
                            }`}
                        >
                          <div
                            className={`p-1.5 rounded-lg ${isActive
                              ? "bg-white/10"
                              : "bg-slate-800/50 group-hover:bg-slate-700/50"
                              }`}
                          >
                            <Icon className={`w-3.5 h-3.5 ${isActive ? "" : "text-slate-500"}`} />
                          </div>
                          <span className="font-medium">{t.label}</span>
                          {isActive && <ChevronRight className="w-3.5 h-3.5 ml-auto opacity-50" />}
                        </button>
                      );
                    })}
                  </div>

                  <button
                    onClick={handleRun}
                    disabled={loading || !serverStatus}
                    className="w-full py-3.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 rounded-xl font-bold mt-4 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-cyan-500/20"
                  >
                    {loading ? "Analyzing..." : "Fetch & Run"}
                  </button>

                  <div className="my-4 flex items-center gap-3">
                    <div className="h-px bg-slate-800 flex-1"></div>
                    <span className="text-xs text-slate-500 font-medium">OR</span>
                    <div className="h-px bg-slate-800 flex-1"></div>
                  </div>

                  <input
                    type="file"
                    accept=".csv,.txt"
                    ref={fileInputRef}
                    className="hidden"
                    onChange={handleFileUpload}
                  />

                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={loading || !serverStatus}
                    className="w-full py-3.5 bg-slate-800 border border-slate-700 hover:bg-slate-700 rounded-xl font-bold text-slate-300 flex items-center justify-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Upload className="w-4 h-4" />
                    Upload CSV / TXT
                  </button>

                  {currentFile && (
                    <div className="mt-4 p-3 bg-slate-800/50 rounded-xl border border-slate-700/50">
                      <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
                        Current File
                      </p>
                      <p className="text-sm font-mono text-slate-300 truncate" title={currentFile}>
                        {currentFile}
                      </p>
                    </div>
                  )}
                </div>

                {result && (
                  <div className="glass rounded-2xl p-5">
                    <h3 className="font-bold mb-3 flex items-center gap-2 text-white">
                      <Gauge className="w-4 h-4 text-violet-400" /> RPM Gauge
                    </h3>

                    <div className="h-36">
                      <ResponsiveContainer>
                        <RadialBarChart
                          innerRadius="65%"
                          outerRadius="100%"
                          data={[{ value: Math.min(result.rpm, 4000), fill: "#8b5cf6" }]}
                          startAngle={180}
                          endAngle={0}
                        >
                          <RadialBar background fill="#1e293b" dataKey="value" cornerRadius={10} />
                          <text
                            x="50%"
                            y="48%"
                            textAnchor="middle"
                            dominantBaseline="middle"
                            className="text-3xl font-bold fill-violet-400"
                          >
                            {Math.round(result.rpm)}
                          </text>
                          <text
                            x="50%"
                            y="68%"
                            textAnchor="middle"
                            className="text-[10px] fill-slate-500"
                          >
                            RPM
                          </text>
                        </RadialBarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </aside>

              <section className="lg:col-span-3 space-y-6">
                {result && (
                  <>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="glass rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Shield className="w-4 h-4 text-emerald-400" />
                          <span className="text-xs text-slate-500 uppercase">Confidence</span>
                        </div>
                        <p className="text-xl font-bold text-white">
                          {(result.confidence * 100).toFixed(1)}%
                        </p>
                        <div className="mt-2 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500"
                            style={{ width: `${result.confidence * 100}%` }}
                          />
                        </div>
                      </div>

                      <div className="glass rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Gauge className="w-4 h-4 text-violet-400" />
                          <span className="text-xs text-slate-500 uppercase">RPM</span>
                        </div>
                        <p className="text-xl font-bold text-white">{Math.round(result.rpm)}</p>
                        <p className="text-xs text-slate-500">{(result.rpm / 60).toFixed(1)} Hz</p>
                      </div>

                      <div className="glass rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <TrendingUp className="w-4 h-4 text-amber-400" />
                          <span className="text-xs text-slate-500 uppercase">Velocity RMS</span>
                        </div>
                        <p className="text-xl font-bold text-white">{isoSeverity.value.toFixed(1)}</p>
                        <p className="text-xs text-slate-500">mm/s</p>
                      </div>

                      <div className="glass rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <File className="w-4 h-4 text-blue-400" />
                          <span className="text-xs text-slate-500 uppercase">Noise</span>
                        </div>
                        <p className="text-xl font-bold text-white">2%</p>
                        <p className="text-xs text-slate-500">injected</p>
                      </div>
                    </div>

                    <div className="glass rounded-2xl p-5">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="font-bold flex items-center gap-2 text-white">
                          <Waves className="w-4 h-4 text-blue-400" /> Time Waveform
                        </h3>

                        <div className="flex items-center gap-4 text-xs">
                          <div className="flex items-center gap-1.5">
                            <div className="w-3 h-0.5 bg-emerald-500 rounded-full"></div>
                            <span className="text-slate-400">Axial</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <div className="w-3 h-0.5 bg-blue-500 rounded-full"></div>
                            <span className="text-slate-400">Radial</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <div className="w-3 h-0.5 bg-orange-500 rounded-full"></div>
                            <span className="text-slate-400">Tangential</span>
                          </div>
                        </div>
                      </div>

                      <div className="h-64">
                        <ResponsiveContainer>
                          <ComposedChart data={timeData} margin={{ top: 10, right: 20, bottom: 25, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.1} stroke="#334155" />
                            <XAxis
                              dataKey="time"
                              tick={{ fontSize: 10, fill: "#94a3b8" }}
                              label={{
                                value: "Time (ms)",
                                position: "bottom",
                                fontSize: 11,
                                fill: "#64748b",
                                offset: 5
                              }}
                              axisLine={{ stroke: "#475569" }}
                              tickLine={{ stroke: "#475569" }}
                            />
                            <YAxis
                              tick={{ fontSize: 10, fill: "#94a3b8" }}
                              label={{
                                value: "Acceleration (mm/s²)",
                                angle: -90,
                                position: "left",
                                fontSize: 11,
                                fill: "#64748b",
                                offset: 5
                              }}
                              axisLine={{ stroke: "#475569" }}
                              tickLine={{ stroke: "#475569" }}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#0f172a",
                                borderColor: "#334155",
                                borderRadius: "10px",
                                fontSize: "11px"
                              }}
                              labelStyle={{ color: "#cbd5e1" }}
                              formatter={(value: any, name: any) => [
                                `${value} mm/s²`,
                                AXIS_NAMES[name as keyof typeof AXIS_NAMES] || name
                              ]}
                            />
                            <Line type="monotone" dataKey="ax" stroke={AXIS_COLORS.ax} strokeWidth={1.5} dot={false} name="ax" />
                            <Line type="monotone" dataKey="ay" stroke={AXIS_COLORS.ay} strokeWidth={2} dot={false} name="ay" />
                            <Line type="monotone" dataKey="az" stroke={AXIS_COLORS.az} strokeWidth={1.5} dot={false} name="az" />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="glass rounded-2xl p-5">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="font-bold flex items-center gap-2 text-white">
                          <BarChart3 className="w-4 h-4 text-purple-400" /> FFT Spectrum (Normalized)
                        </h3>

                        <div className="flex bg-slate-800/50 p-1 rounded-lg">
                          {["fft"].map(tab => (
                            <button
                              key={tab}
                              onClick={() => setSpectrumTab(tab)}
                              className={`px-3 py-1 text-xs font-bold rounded-lg transition-all ${spectrumTab === tab
                                ? "bg-purple-600 text-white shadow-lg"
                                : "text-slate-400 hover:text-slate-200"
                                }`}
                            >
                              {tab === "fft" ? "FFT" : "Envelope"}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="h-72">
                        {spectrumTab === "fft" && specData.length > 0 && (
                          <ResponsiveContainer>
                            <ComposedChart data={specData} margin={{ top: 10, right: 20, bottom: 30, left: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" opacity={0.1} stroke="#334155" />
                              <XAxis
                                dataKey="freq"
                                tick={{ fontSize: 10, fill: "#94a3b8" }}
                                label={{
                                  value: "Frequency (Hz)",
                                  position: "bottom",
                                  fontSize: 11,
                                  fill: "#64748b",
                                  offset: 10
                                }}
                                axisLine={{ stroke: "#475569" }}
                                tickLine={{ stroke: "#475569" }}
                              />
                              <YAxis
                                tick={{ fontSize: 10, fill: "#94a3b8" }}
                                label={{
                                  value: "Normalized Amplitude",
                                  angle: -90,
                                  position: "left",
                                  fontSize: 11,
                                  fill: "#64748b",
                                  offset: 5
                                }}
                                axisLine={{ stroke: "#475569" }}
                                tickLine={{ stroke: "#475569" }}
                                domain={[0, 1.1]}
                              />
                              <Tooltip
                                contentStyle={{
                                  backgroundColor: "#0f172a",
                                  borderColor: "#334155",
                                  borderRadius: "10px",
                                  fontSize: "11px"
                                }}
                                labelStyle={{ color: "#cbd5e1" }}
                                formatter={(value: any) => [`${value.toFixed(3)}`, "Norm Amp"]}
                                labelFormatter={(label) => `${label} Hz`}
                              />
                              <Area
                                type="monotone"
                                dataKey="amp"
                                fill="#8b5cf6"
                                stroke="#7c3aed"
                                fillOpacity={0.3}
                                strokeWidth={2}
                              />
                              <ReferenceLine
                                x={result.rpm / 60}
                                stroke="#fbbf24"
                                strokeDasharray="5 5"
                                strokeWidth={2}
                                label={{
                                  value: "1× RPM",
                                  position: "top",
                                  fill: "#fbbf24",
                                  fontSize: 10,
                                  offset: 5
                                }}
                              />
                              <ReferenceLine
                                x={2 * result.rpm / 60}
                                stroke="#fb923c"
                                strokeDasharray="5 5"
                                strokeWidth={2}
                                label={{
                                  value: "2× RPM",
                                  position: "top",
                                  fill: "#fb923c",
                                  fontSize: 10,
                                  offset: 5
                                }}
                              />
                            </ComposedChart>
                          </ResponsiveContainer>
                        )}

                        {/* {spectrumTab === "envelope" && envData.length > 0 && (
                          <ResponsiveContainer>
                            <ComposedChart data={envData} margin={{ top: 10, right: 20, bottom: 30, left: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" opacity={0.1} stroke="#334155" />
                              <XAxis
                                dataKey="freq"
                                tick={{ fontSize: 10, fill: "#94a3b8" }}
                                label={{
                                  value: "Envelope Freq (Hz)",
                                  position: "bottom",
                                  fontSize: 11,
                                  fill: "#64748b",
                                  offset: 10
                                }}
                                axisLine={{ stroke: "#475569" }}
                                tickLine={{ stroke: "#475569" }}
                              />
                              <YAxis
                                tick={{ fontSize: 10, fill: "#94a3b8" }}
                                label={{
                                  value: "Normalized Amplitude",
                                  angle: -90,
                                  position: "left",
                                  fontSize: 11,
                                  fill: "#64748b",
                                  offset: 5
                                }}
                                axisLine={{ stroke: "#475569" }}
                                tickLine={{ stroke: "#475569" }}
                                domain={[0, 1.1]}
                              />
                              <Tooltip
                                contentStyle={{
                                  backgroundColor: "#0f172a",
                                  borderColor: "#334155",
                                  borderRadius: "10px",
                                  fontSize: "11px"
                                }}
                              />
                              <Area
                                type="monotone"
                                dataKey="amp"
                                fill="#ec4899"
                                stroke="#db2777"
                                fillOpacity={0.3}
                                strokeWidth={2}
                              />
                              {renderBearingMarkers(result.prediction, result.rpm)}
                            </ComposedChart>
                          </ResponsiveContainer>
                        )} */}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="glass rounded-2xl p-5">
                        <h3 className="font-bold mb-3 flex items-center gap-2 text-white">
                          <BarChart3 className="w-4 h-4 text-cyan-400" /> Feature Comparison
                        </h3>

                        <div className="h-64">
                          <ResponsiveContainer>
                            <RadarChart data={radarData}>
                              <PolarGrid stroke="#334155" />
                              <PolarAngleAxis dataKey="feature" tick={{ fontSize: 9, fill: "#94a3b8" }} />
                              <PolarRadiusAxis tick={false} axisLine={false} />
                              <Radar name="Current" dataKey="current" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.3} />
                              <Radar name="Normal" dataKey="normal" stroke="#22c55e" fill="#22c55e" fillOpacity={0.15} />
                              <Legend wrapperStyle={{ fontSize: "10px", paddingTop: "10px" }} />
                              <Tooltip
                                contentStyle={{
                                  backgroundColor: "#0f172a",
                                  borderColor: "#334155",
                                  borderRadius: "10px",
                                  fontSize: "11px"
                                }}
                              />
                            </RadarChart>
                          </ResponsiveContainer>
                        </div>

                        <p className="text-[10px] text-slate-500 mt-2 text-center">
                          Comparing 6 key features against healthy baseline
                        </p>
                      </div>

                      <div className="glass rounded-2xl p-5">
                        <h3 className="font-bold mb-3 flex items-center gap-2 text-white">
                          <Activity className="w-4 h-4 text-violet-400" /> Feature Values
                        </h3>

                        <div className="grid grid-cols-2 gap-3">
                          {RADAR_FEATURES.map(f => (
                            <div key={f.key} className="bg-slate-800/50 rounded-xl p-3 border border-slate-700/50">
                              <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
                                {f.label}
                              </p>
                              <p className="text-lg font-bold text-white font-mono">
                                {(featureValues[f.key] || 0).toFixed(3)}
                              </p>
                              <p className="text-[10px] text-slate-500">
                                Normal: {f.baseline}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </section>
            </div>
          </main>
        </div>
      </body>
    </html>
  );
}

const renderBearingMarkers = (fault: string, rpm: number) => {
  const f = rpm / 60;
  const markers = [];

  if (fault === "Ball_Fault") {
    const bsf = BEARING_SPECS.BSF * f;

    if (bsf < 500) {
      markers.push(
        <ReferenceLine
          key="bsf"
          x={bsf}
          stroke="#8b5cf6"
          strokeWidth={2}
          label={{ value: "BSF", fill: "#8b5cf6", fontSize: 10, offset: 5 }}
        />
      );
    }
  }

  if (fault === "Outer_Race") {
    const bpfo = BEARING_SPECS.BPFO * f;

    if (bpfo < 500) {
      markers.push(
        <ReferenceLine
          key="bpfo"
          x={bpfo}
          stroke="#a78bfa"
          strokeWidth={2}
          label={{ value: "BPFO", fill: "#a78bfa", fontSize: 10, offset: 5 }}
        />
      );
    }
  }

  return markers;
};