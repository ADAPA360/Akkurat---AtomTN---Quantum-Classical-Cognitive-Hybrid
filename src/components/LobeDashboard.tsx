import { TelemetryPayload } from "../types";
import { BrainCircuit, Activity, Cpu, Network, AudioWaveform } from "lucide-react";
import { motion } from "motion/react";

const LOBE_CONFIG = {
  sensory: { label: "Sensory", color: "bg-blue-500", icon: AudioWaveform },
  memory: { label: "Memory", color: "bg-emerald-500", icon: BrainCircuit },
  semantic: { label: "Semantic", color: "bg-purple-500", icon: Network },
  planning: { label: "Planning", color: "bg-amber-500", icon: Cpu },
  regulation: { label: "Regulation", color: "bg-rose-500", icon: Activity },
};

export default function LobeDashboard({ data }: { data: TelemetryPayload | null }) {
  if (!data) return <div className="p-4 text-slate-500 font-mono text-xs">Waiting for telemetry...</div>;

  return (
    <div className="flex flex-col h-full bg-slate-950/80 backdrop-blur border-r border-slate-800 p-6 overflow-hidden">
      <div className="mb-8">
        <h2 className="text-xl font-medium tracking-tight text-white mb-2">Cognitive Cortex</h2>
        <div className="flex items-center space-x-2 text-xs font-mono text-slate-400">
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_10px_#10b981]" />
          <span>SYS_TICK: {data.tick}</span>
        </div>
      </div>

      <div className="flex-1 space-y-6">
        {Object.entries(LOBE_CONFIG).map(([key, config]) => {
          const ldata = data.lobes[key as keyof typeof data.lobes];
          const loadPercent = Math.min(100, Math.max(0, (ldata?.norm || 0) * 15)).toFixed(0);
          const Icon = config.icon;

          return (
            <div key={key} className="relative group">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3 text-slate-200">
                  <div className={`p-1.5 rounded-md bg-slate-900 border border-slate-700 group-hover:border-slate-500 transition-colors`}>
                    <Icon size={14} className="text-slate-300" />
                  </div>
                  <span className="font-medium tracking-wider text-xs uppercase">{config.label}</span>
                </div>
                <div className="font-mono text-xs text-slate-400">{loadPercent}%</div>
              </div>

              {/* Progress track */}
              <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden relative">
                <motion.div
                  className={`absolute top-0 left-0 bottom-0 ${config.color}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${loadPercent}%` }}
                  transition={{ type: "spring", bounce: 0, duration: 0.8 }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-8 pt-6 border-t border-slate-800">
        <h3 className="text-xs uppercase tracking-widest text-slate-500 font-semibold mb-4">AtomTN Reservoir</h3>
        
        <div className="space-y-3 font-mono text-[10px]">
          <div className="flex justify-between items-center text-slate-400">
            <span>Global L2 Norm:</span>
            <span className="text-slate-200">{Number(data.global_state_norm).toFixed(4)}</span>
          </div>
          <div className="flex justify-between items-center text-slate-400">
            <span>Holonomy Trace:</span>
            <span className="text-slate-200">{Number(data.physics.holonomy_trace).toFixed(4)}</span>
          </div>
          <div className="flex justify-between items-center text-slate-400">
            <span>Spacetime Entropy:</span>
            <span className="text-slate-200">{Number(data.physics.entropy).toFixed(4)}</span>
          </div>
          <div className="flex justify-between items-center text-slate-400">
            <span>Signal Coherence:</span>
            <motion.span 
              className="text-emerald-400"
              key={data.bus.mean_phase_coherence}
              initial={{ opacity: 0.5 }}
              animate={{ opacity: 1 }}
            >
              {Number(data.bus.mean_phase_coherence).toFixed(4)}
            </motion.span>
          </div>
        </div>
      </div>
    </div>
  );
}
