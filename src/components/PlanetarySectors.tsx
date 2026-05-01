import { motion } from "motion/react";
import { Truck, FlaskConical, Globe, TrendingUp } from "lucide-react";
import { TelemetryPayload } from "../types";

export default function PlanetarySectors({ data }: { data: TelemetryPayload | null }) {
  if (!data) return null;

  // Derive metrics from the real telemetry (or simulated physics)
  const logisticsEfficiency = (100 - Number(data.physics.entropy) * 10).toFixed(1);
  const discoveryRate = (Number(data.physics.holonomy_trace) * 5.2).toFixed(1);
  const gridStability = (Number(data.bus.mean_phase_coherence) * 100).toFixed(1);
  const financialValue = (Number(data.global_state_norm) * 1.83).toFixed(2);

  return (
    <div className="absolute bottom-6 left-6 right-6 z-10 flex gap-4 pointer-events-none">
      
      {/* Logistics & Supply Chain */}
      <motion.div 
        className="flex-1 bg-slate-900/60 backdrop-blur-md border border-slate-800 rounded-lg p-4 pointer-events-auto"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="flex items-center gap-2 text-blue-500 mb-2">
          <Truck size={16} />
          <h3 className="text-xs font-bold uppercase tracking-wider">Supply Chain</h3>
        </div>
        <p className="text-[10px] text-slate-400 mb-3 leads-tight">
          TetraMesh64 active rerouting via Google Maps & Cloud IoT. Geodesic turbulence adaptation.
        </p>
        <div className="text-xl font-mono text-blue-400">
          {logisticsEfficiency}% <span className="text-[10px] text-slate-500 uppercase">Efficiency</span>
        </div>
      </motion.div>

      {/* Scientific Discovery */}
      <motion.div 
        className="flex-1 bg-slate-900/60 backdrop-blur-md border border-slate-800 rounded-lg p-4 pointer-events-auto"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="flex items-center gap-2 text-rose-500 mb-2">
          <FlaskConical size={16} />
          <h3 className="text-xs font-bold uppercase tracking-wider">Scientific Oracles</h3>
        </div>
        <p className="text-[10px] text-slate-400 mb-3 leads-tight">
          Hamiltonian molecular binding simulation on AtomTN via Gemini & Google Scholar.
        </p>
        <div className="text-xl font-mono text-rose-400">
          {discoveryRate}x <span className="text-[10px] text-slate-500 uppercase">Accel</span>
        </div>
      </motion.div>

      {/* Planetary Governance */}
      <motion.div 
        className="flex-1 bg-slate-900/60 backdrop-blur-md border border-slate-800 rounded-lg p-4 pointer-events-auto"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <div className="flex items-center gap-2 text-emerald-500 mb-2">
          <Globe size={16} />
          <h3 className="text-xs font-bold uppercase tracking-wider">Global Homeostasis</h3>
        </div>
        <p className="text-[10px] text-slate-400 mb-3 leads-tight">
          TreeTensorNetwork regulation of Earth Engine state vs BigQuery thermal output.
        </p>
        <div className="text-xl font-mono text-emerald-400">
          {gridStability}% <span className="text-[10px] text-slate-500 uppercase">Stability</span>
        </div>
      </motion.div>

      {/* Economic Value */}
      <motion.div 
        className="flex-shrink-0 w-48 bg-purple-900/40 backdrop-blur-md border border-purple-800/50 rounded-lg p-4 pointer-events-auto flex flex-col justify-center align-middle relative overflow-hidden"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.4 }}
      >
        <div className="absolute top-0 right-0 p-2 opacity-20">
            <TrendingUp size={48} />
        </div>
        <div className="text-[10px] font-bold uppercase tracking-widest text-purple-300 mb-1 z-10">
          Projected Value
        </div>
        <div className="text-2xl font-mono text-white z-10">
          ${financialValue}B
        </div>
      </motion.div>

    </div>
  );
}
