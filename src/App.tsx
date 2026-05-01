import React, { useEffect, useState, useRef } from "react";
import ASINetwork from "./components/ASINetwork";
import LobeDashboard from "./components/LobeDashboard";
import PlanetarySectors from "./components/PlanetarySectors";
import { TelemetryPayload } from "./types";
import { Terminal, Cpu, Database, Command, Settings, Key, X } from "lucide-react";
import { motion } from "motion/react";

export default function App() {
  const [telemetry, setTelemetry] = useState<TelemetryPayload | null>(null);
  const [intentInput, setIntentInput] = useState("");
  const [apiKey, setApiKey] = useState(localStorage.getItem('gemini_api_key') || '');
  const [showSettings, setShowSettings] = useState(false);
  const [isInjecting, setIsInjecting] = useState(false);
  const [logs, setLogs] = useState<string[]>([
    "[SYSTEM] Akkurat / AtomTN Digital Twin Initialized.",
    "[SYSTEM] TetraMesh64 Substrate mapped to Google Cloud infrastructure.",
    "[SYSTEM] Waiting for intent injection to awake ASI nodes..."
  ]);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const eventSource = new EventSource("/api/telemetry");

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setTelemetry(data);
      } catch (e) {
        console.error("Telemetry parsing error", e);
      }
    };

    return () => {
      eventSource.close();
    };
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const addLog = (msg: string) => {
    setLogs((prev) => [...prev, `[${new Date().toISOString().split("T")[1].slice(0, -1)}] ${msg}`]);
  };

  const handleInjectIntent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!intentInput.trim() || isInjecting) return;

    const intent = intentInput;
    setIntentInput("");
    setIsInjecting(true);
    addLog(`> INTENT INJECTED: ${intent}`);
    addLog("[REGULATION] Analyzing intent topology...");

    try {
      const res = await fetch("/api/intent", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "x-gemini-api-key": apiKey
        },
        body: JSON.stringify({ intent })
      });
      const data = await res.json();
      
      if (data.error) {
        addLog(`[ERROR] ${data.error}`);
      } else {
        addLog(`[PLANNING] Calculated topological drift: ${data.topological_drift}`);
        addLog(`[EXECUTING] Plan: ${data.plan}`);
        addLog(`[ORACLES] Activating ASI Nodes: ${data.activated_nodes.join(", ")}`);
      }
    } catch (err) {
      addLog(`[ERROR] Failed to reach Cortex module: ${err}`);
    } finally {
      setIsInjecting(false);
    }
  };

  return (
    <div className="h-screen w-full flex bg-slate-950 text-slate-200 font-sans selection:bg-purple-900 overflow-hidden">
      {/* Left Sidebar: Cognitive Lobes */}
      <div className="w-72 h-full shrink-0 z-10">
        <LobeDashboard data={telemetry} />
      </div>

      {/* Center: AGI Core & ASI Networks */}
      <div className="flex-1 relative h-full flex flex-col items-center justify-center border-r border-slate-800 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-slate-950">
        
        {/* Subtle grid background */}
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCI+PHBhdGggZD0iTTAgMGg0MHY0MEgweiIgZmlsbD0ibm9uZSIvPPHBhdGggZD0iTTAgNDBMNDAgMCIgc3Ryb2tlPSJyZ2JhKDI1NSwyNTUsMjU1LDAuMDMpIiBzdHJva2Utd2lkdGg9IjEiLz48L3N2Zz4=')] opacity-30 pointer-events-none" />

        <div className="absolute top-6 left-6 z-10 flex items-center space-x-2 text-slate-400">
          <Database size={16} />
          <span className="text-xs font-mono font-medium tracking-widest uppercase">Planetary Symbiosis Graph</span>
        </div>

        {telemetry && (
          <div className="w-full max-w-2xl aspect-square absolute">
            <ASINetwork nodes={telemetry.asi_nodes} drift={telemetry.physics.entropy} />
          </div>
        )}

        <PlanetarySectors data={telemetry} />
      </div>

      {/* Right Sidebar: Terminal & Controls */}
      <div className="w-96 flex flex-col shrink-0 h-full z-10 bg-slate-900/50 backdrop-blur">
        <div className="p-4 border-b border-slate-800 bg-slate-950/80 flex items-start justify-between">
          <div>
            <div className="flex items-center space-x-2 mb-1 text-purple-400">
              <Command size={16} />
              <h3 className="text-sm font-bold tracking-widest uppercase">Orchestration</h3>
            </div>
            <p className="text-[10px] uppercase font-mono text-slate-500">Inject natural language to bend the quantum manifold and command ASI execution.</p>
          </div>
          <button 
            onClick={() => setShowSettings(true)}
            className="p-1.5 text-slate-500 hover:text-slate-300 hover:bg-slate-800 rounded transition-colors"
            title="App Settings"
          >
            <Settings size={16} />
          </button>
        </div>

        <div className="flex-1 p-4 font-mono text-[10px] leading-relaxed text-slate-300 overflow-y-auto bg-[#0a0a0f]">
          {logs.map((log, i) => (
            <motion.div 
              initial={{ opacity: 0, x: -10 }} 
              animate={{ opacity: 1, x: 0 }} 
              key={i} 
              className={`mb-2 ${log.startsWith(">") ? 'text-amber-400' : log.includes("[ERROR]") ? 'text-rose-500' : log.includes("[ORACLE") ? 'text-emerald-400' : 'text-slate-400'}`}
            >
              {log}
            </motion.div>
          ))}
          <div ref={logEndRef} />
        </div>

        <div className="p-4 border-t border-slate-800 bg-slate-950 relative">
          {isInjecting && (
             <div className="absolute top-0 left-0 right-0 h-0.5 bg-slate-800 overflow-hidden">
                <motion.div 
                  className="h-full bg-purple-500" 
                  initial={{ x: "-100%" }}
                  animate={{ x: "100%" }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                />
             </div>
          )}
          <form onSubmit={handleInjectIntent} className="flex flex-col space-y-2">
            <label className="text-[10px] font-mono uppercase text-slate-500 tracking-wider">Intent Override Vector</label>
            <div className="flex items-stretch bg-black border border-slate-700 rounded-md overflow-hidden focus-within:border-purple-500 focus-within:ring-1 focus-within:ring-purple-500/50 transition-all">
              <div className="pl-3 pr-2 flex items-center text-slate-500">
                <Terminal size={14} />
              </div>
              <input 
                type="text" 
                value={intentInput}
                onChange={(e) => setIntentInput(e.target.value)}
                placeholder="e.g. Synthesize protein structures for Mars..." 
                className="flex-1 bg-transparent border-none py-2.5 text-xs text-slate-200 focus:outline-none focus:ring-0 placeholder:text-slate-600"
                disabled={isInjecting}
              />
            </div>
          </form>
        </div>
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="absolute inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-slate-950 border border-slate-800 rounded-lg shadow-2xl w-full max-w-md overflow-hidden flex flex-col"
          >
            <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
              <div className="flex items-center space-x-2 text-slate-200">
                <Settings size={16} />
                <h2 className="text-sm font-semibold">Application Settings</h2>
              </div>
              <button 
                onClick={() => setShowSettings(false)}
                className="text-slate-500 hover:text-slate-300 transition-colors"
              >
                <X size={16} />
              </button>
            </div>
            
            <div className="p-6 space-y-4">
              <div className="space-y-2">
                <label className="text-xs font-medium text-slate-300 flex items-center space-x-2">
                  <Key size={14} className="text-purple-400" />
                  <span>Gemini API Key</span>
                </label>
                <input 
                  type="password" 
                  value={apiKey}
                  onChange={(e) => {
                    setApiKey(e.target.value);
                    localStorage.setItem('gemini_api_key', e.target.value);
                  }}
                  placeholder="AIzaSy..." 
                  className="w-full bg-slate-900 border border-slate-700 rounded-md px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-1 focus:ring-purple-500"
                />
                <p className="text-[10px] text-slate-500 mt-1">
                  Required. Your API key is stored only in your local browser and sent to the application backend to invoke models.
                </p>
              </div>
            </div>
            
            <div className="p-4 border-t border-slate-800 bg-slate-900/50 flex justify-end">
              <button 
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-md text-sm font-medium transition-colors"
              >
                Done
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
