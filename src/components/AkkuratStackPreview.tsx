import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { FileCode2, X, Layers, Box, Fingerprint, Activity, Network, Binary, Shapes, Info, Atom, Zap, Maximize, Cpu, Globe, Crosshair, Sigma, ArrowRightLeft, Anchor, Variable, AlignCenter, Tally3, Infinity } from 'lucide-react';

const CATEGORIES = [
  {
    name: "Akkurat Architecture",
    scripts: [
      {
        id: "akkurat_atom_hybrid.py",
        title: "Akkurat / AtomTN Hybrid Runtime",
        icon: <Layers size={18} />,
        description: "Production-ready orchestrator that fuses AtomTN neuromorphic reservoir features, a lightweight tensorized/Numpy CfC-style controller, and optional Akkurat digital twin governance updates.",
        features: ["Import-safe initialization", "Dynamic attachment of AtomTN reservoirs", "Seamless fusion architectures (gated, additive, etc.)"],
        color: "from-blue-500/20 text-blue-400 border-blue-500/30"
      },
      {
        id: "atom_adapter_runtime.py",
        title: "AtomTN Adapter Runtime",
        icon: <Box size={18} />,
        description: "Production-oriented bridge between the AtomTN neuromorphic reservoir family and Akkurat's governed digital twin kernel.",
        features: ["CPU-safe fallback support", "Digital Twin mapping support", "Explicit degradation telemetry"],
        color: "from-cyan-500/20 text-cyan-400 border-cyan-500/30"
      },
      {
        id: "cognitive_lobe_runtime.py",
        title: "Cognitive Lobe Runtime",
        icon: <Fingerprint size={18} />,
        description: "Production composition layer for the Akkurat cognitive architecture. Establishes the 5 native neural lobes over signal buses.",
        features: ["Phase-synchronous signal routing", "Assembly feedback and consolidation", "Sensory, Memory, Semantic, Planning, and Regulation domains"],
        color: "from-purple-500/20 text-purple-400 border-purple-500/30"
      },
      {
        id: "digital_twin_kernel.py",
        title: "Governed Digital Twin Kernel",
        icon: <Network size={18} />,
        description: "Production CPU-first digital-twin substrate. Manages a fixed-width latent-state tree for physical, virtual, data, cognitive, and physics nodes.",
        features: ["Heterogeneous deterministic projections", "Ring-buffer sketch histories & Causal probing", "Governed action execution pipelines"],
        color: "from-emerald-500/20 text-emerald-400 border-emerald-500/30"
      },
      {
        id: "ncp.py",
        title: "Neural Circuit Policy (NCP)",
        icon: <Activity size={18} />,
        description: "Shared bounded recurrent substrate for cognitive lobes with continuous-time hidden dynamics and adaptive time constants.",
        features: ["TensorTrain recurrent backbone support", "Stable numeric vector inputs", "State serialization & full checkpointing"],
        color: "from-rose-500/20 text-rose-400 border-rose-500/30"
      },
      {
        id: "cfc.py",
        title: "Tensorized CfC Policy",
        icon: <Shapes size={18} />,
        description: "Continuous-time Neural dynamics substrate utilizing multi-cell Tensorized Closed-form Continuous-time policies.",
        features: ["TensorTrain/MPO-backed cell backbones", "Deterministic seeded initialization", "Bounded state updates"],
        color: "from-amber-500/20 text-amber-400 border-amber-500/30"
      }
    ]
  },
  {
    name: "AtomTN: Runtime & Policies",
    scripts: [
      {
        id: "atom.py",
        title: "Atom Runtime Wrapper",
        icon: <Atom size={18} />,
        description: "AtomTN atom wrapper and demo runner. Wires together the core AtomTN substrate including geometry, TTNState, vibration, and flow simulation.",
        features: ["Geometry Setup", "Vibration Attachment", "Flow & Evolution Orchestration"],
        color: "from-fuchsia-500/20 text-fuchsia-400 border-fuchsia-500/30"
      },
      {
        id: "neuromorphic.py",
        title: "Neuromorphic Substrate",
        icon: <Zap size={18} />,
        description: "Holographic Liquid State Machine (HLSM) powered by AtomTN. Turns the AtomTN physics engine into a production-friendly reservoir computing substrate.",
        features: ["Multiple execution profiles", "Geometry Vector Encoding", "Holographic Readout API"],
        color: "from-violet-500/20 text-violet-400 border-violet-500/30"
      },
      {
        id: "atom_ncp.py",
        title: "AtomTN Quantum NCP",
        icon: <Activity size={18} />,
        description: "Production-compatible Quantum Neural Circuit Policy backed by the AtomTN neuromorphic reservoir runtime.",
        features: ["Sparse Deterministic Encoder", "Hamiltonian TTN Evolution", "Trainable Operator Readout"],
        color: "from-rose-500/20 text-rose-400 border-rose-500/30"
      },
      {
        id: "atom_cfc.py",
        title: "AtomTN Quantum CfC",
        icon: <Shapes size={18} />,
        description: "Production-compatible Quantum Closed-form Continuous-time Policy backed by the shared AtomTN reservoir.",
        features: ["Coupled Quantum & Classical States", "Holographic Feature Extraction", "Multi-Cell CfC Dynamics"],
        color: "from-amber-500/20 text-amber-400 border-amber-500/30"
      }
    ]
  },
  {
    name: "AtomTN: Tensor Operations",
    scripts: [
      {
        id: "ttn_state.py",
        title: "TTN State Container",
        icon: <Network size={18} />,
        description: "Production TTN state container and contraction substrate. Handles validated Tree Tensor Network state storage and basic tensor operations.",
        features: ["Top-down & Bottom-up Environments", "Parent-bond SVD truncation", "QR Canonicalization"],
        color: "from-blue-500/20 text-blue-400 border-blue-500/30"
      },
      {
        id: "apply.py",
        title: "Hamiltonian Application",
        icon: <ArrowRightLeft size={18} />,
        description: "Applies AtomTN Hamiltonian/operator containers to a TTNState. Supports fast scaffold approximation or exact direct-sum zero-up application.",
        features: ["Scaffold apply path", "Direct-sum zip-up baseline", "Tree operator accumulation"],
        color: "from-cyan-500/20 text-cyan-400 border-cyan-500/30"
      },
      {
        id: "hamiltonian.py",
        title: "Hamiltonian Builder",
        icon: <Sigma size={18} />,
        description: "Converts AtomTN flow, vibration, projection, and local-fiber context into Hamiltonian term containers.",
        features: ["TreeMPO Fallback", "CompiledTreeOperator structure", "LCA grouping"],
        color: "from-emerald-500/20 text-emerald-400 border-emerald-500/30"
      },
      {
        id: "evolve.py",
        title: "TTN Time Evolver",
        icon: <Maximize size={18} />,
        description: "AtomTN TTN evolution runtime. Implements multiple integration schemes like Euler, RK4, and Lie-Trotter splittings.",
        features: ["RK4 + Zip-up Integration", "Lie-Trotter splitting", "Padded derivative projection"],
        color: "from-lime-500/20 text-lime-400 border-lime-500/30"
      },
      {
        id: "tn.py",
        title: "Unified Tensor Network Lib",
        icon: <Binary size={18} />,
        description: "Production-ready CPU-first TT/MPO + Tucker runtime. Includes randomized SVDs, block TSQR orthogonalization, and CuPy GPU support.",
        features: ["TT/MPO + Tucker core classes", "Approximate TT-SVD routines", "CuPy GPU acceleration hooks"],
        color: "from-slate-500/20 text-slate-400 border-slate-500/30"
      }
    ]
  },
  {
    name: "AtomTN: Geometry & Physics",
    scripts: [
      {
        id: "flow.py",
        title: "Geodesic Flow Solvers",
        icon: <Globe size={18} />,
        description: "Flow solvers and diagnostics. Transport layer offering a deterministic graph-flow substrate for both scalar and matrix-valued edge fields.",
        features: ["Scalar Hydrodynamic Flow", "Noncommutative SU(2) Flow", "Divergence Diagnostics"],
        color: "from-teal-500/20 text-teal-400 border-teal-500/30"
      },
      {
        id: "geometry.py",
        title: "Geometry Scaffold",
        icon: <Box size={18} />,
        description: "Deterministic geometry substrate and tree compilation. Owns the base TetraMesh64 object and basic GraphCalculus operations.",
        features: ["K-ary Tree Compilation", "Morton/Z-Order Spatial Grouping", "Commutative Graph Laplacian"],
        color: "from-indigo-500/20 text-indigo-400 border-indigo-500/30"
      },
      {
        id: "vibration.py",
        title: "Vibration / Phonon Bath",
        icon: <Activity size={18} />,
        description: "Deterministic frequency grids and spectral-density matching. Manages discrete vibration baths for quantum subsystems.",
        features: ["Fractal & Fibonacci Grids", "Ohmic/Powerlaw Spectral Densities", "Thermal Envelope Couplings"],
        color: "from-red-500/20 text-red-400 border-red-500/30"
      }
    ]
  },
  {
    name: "AtomTN: Noncommutative",
    scripts: [
      {
        id: "fuzzy_backend.py",
        title: "Fuzzy SU(2) Backend",
        icon: <Variable size={18} />,
        description: "Matrix-valued calculus on a graph. Handles Spin-l generators, twisted reality mappings, and SU(2) geometric structures.",
        features: ["Spin-l Generators", "Twisted Reality Unitaries", "Higher-harmonic Basis Search"],
        color: "from-pink-500/20 text-pink-400 border-pink-500/30"
      },
      {
        id: "projection.py",
        title: "Holographic Projection",
        icon: <AlignCenter size={18} />,
        description: "Fuzzy k -> local d holographic compression. Converts fuzzy operators to local Hilbert spaces via static and energy-gauge projectors.",
        features: ["Eigen-basis Alignment (Procrustes)", "StepKey-aware Caches", "Frozen Substep Re-Use"],
        color: "from-orange-500/20 text-orange-400 border-orange-500/30"
      },
      {
        id: "holonomy.py",
        title: "Adjoint Transport Holonomy",
        icon: <Anchor size={18} />,
        description: "Converts matrix-valued NC edge flows into stable SO(3) adjoint-frame rotations. Supports Hamiltonian holonomy-coupled edges.",
        features: ["Complex Matrix Exponentials", "SO(3) Rotation Projections", "Deterministic Substep Caches"],
        color: "from-yellow-500/20 text-yellow-400 border-yellow-500/30"
      },
      {
        id: "fiber.py",
        title: "Local Fiber Builder",
        icon: <Tally3 size={18} />,
        description: "Local Hilbert-space dimension policy. Controls static and adaptive bounds and handles local operator bases.",
        features: ["Dynamic Fiber Dimension Scaling", "Operator Basis Verification", "Projection Constraints Filtering"],
        color: "from-emerald-500/20 text-emerald-400 border-emerald-500/30"
      }
    ]
  },
  {
    name: "AtomTN: Math & Specs",
    scripts: [
      {
        id: "schedules.py",
        title: "Generic Schedules",
        icon: <Crosshair size={18} />,
        description: "Score-agnostic scheduling utilities. Converts scalars into bounded integer schedules used for TTN truncation and dimensions.",
        features: ["Score Normalization (Minmax/ZScore)", "EMA Smoothing", "TTN Truncation Caps"],
        color: "from-sky-500/20 text-sky-400 border-sky-500/30"
      },
      {
        id: "curvature.py",
        title: "Curvature Diagnostics",
        icon: <Globe size={18} />,
        description: "Physics-specific score construction. Converts flows and FlowDiagnostics into finite curvature proxies.",
        features: ["Scalar Divergence Scoring", "Su2 Generator Magnitudes", "Hotspot Detection"],
        color: "from-indigo-500/20 text-indigo-400 border-indigo-500/30"
      },
      {
        id: "environments.py",
        title: "TTN Environments",
        icon: <Layers size={18} />,
        description: "Cached TTN environments and messages. Pre-computation structure for fast correct apply (Phase 4 substrate).",
        features: ["Bottom-up Norm Messages", "Operator-inserted Direct Paths", "Dirty Status Tracking"],
        color: "from-purple-500/20 text-purple-400 border-purple-500/30"
      },
      {
        id: "constraints.py",
        title: "Operator Constraints",
        icon: <Info size={18} />,
        description: "Lightweight, seeded operator-basis generation and embedding for abstract local fibers.",
        features: ["Pauli Group Embroidery", "Hermitian Random Matrices", "Adinkra Substrate Constraints"],
        color: "from-slate-500/20 text-slate-400 border-slate-500/30"
      },
      {
        id: "math_utils.py",
        title: "Numerical Utilities",
        icon: <Infinity size={18} />,
        description: "CPU/NumPy-first stable primitives. Finite-value guards, matrix exponential fallbacks, eigensolver wrappers, and SVD accounting.",
        features: ["Hermitization Casts", "Stable SVD Truncation Limits", "Small Matrix Exp Fallbacks"],
        color: "from-zinc-500/20 text-zinc-400 border-zinc-500/30"
      }
    ]
  }
];

export default function AkkuratStackPreview({ onClose }: { onClose: () => void }) {
  const [activeScript, setActiveScript] = useState(CATEGORIES[0].scripts[0].id);
  
  const current = CATEGORIES.flatMap(c => c.scripts).find(s => s.id === activeScript)!;

  return (
    <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-md z-50 flex items-center justify-center p-4 sm:p-8 md:p-12">
      <motion.div 
        initial={{ opacity: 0, scale: 0.95, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 10 }}
        className="w-full max-w-5xl h-full max-h-[800px] bg-slate-900 border border-slate-700/50 rounded-2xl shadow-2xl flex flex-col overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/50">
          <div className="flex items-center gap-3 text-slate-200">
            <FileCode2 className="text-purple-400" />
            <div>
              <h2 className="text-lg font-bold tracking-tight">The Chimera Protocol Stack</h2>
              <p className="text-[10px] uppercase font-mono text-slate-500 tracking-wider">Akkurat Framework & AtomTN Substrate</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar */}
          <div className="w-1/3 min-w-[280px] max-w-[340px] bg-slate-950/50 border-r border-slate-800/50 overflow-y-auto p-4 flex flex-col gap-6">
            {CATEGORIES.map((category, idx) => (
              <div key={idx} className="flex flex-col gap-2">
                <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider px-2 border-b border-slate-800/50 pb-1 mb-1">
                  {category.name}
                </div>
                {category.scripts.map((script) => (
                  <button
                    key={script.id}
                    onClick={() => setActiveScript(script.id)}
                    className={`flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 text-left border ${
                      activeScript === script.id 
                        ? `bg-slate-900 border-slate-700 shadow-sm shadow-black/20` 
                        : 'border-transparent text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                    }`}
                  >
                    <div className={`p-1.5 rounded-md ${activeScript === script.id ? `bg-gradient-to-br ${script.color} text-white` : 'bg-slate-900'}`}>
                      {script.icon}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className={`text-sm font-semibold truncate leading-tight ${activeScript === script.id ? 'text-white' : ''}`}>
                        {script.title}
                      </div>
                      <div className="text-[10px] font-mono opacity-60 truncate mt-0.5">{script.id}</div>
                    </div>
                  </button>
                ))}
              </div>
            ))}
          </div>

          {/* Main Visualizer */}
          <div className="flex-1 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-slate-800/20 via-slate-900 to-slate-900 overflow-y-auto relative">
            <AnimatePresence mode="wait">
              <motion.div
                key={current.id}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
                className="p-8 md:p-12 lg:p-16 h-full flex flex-col"
              >
                <div className="flex items-center gap-4 mb-6">
                  <div className={`p-5 rounded-2xl bg-gradient-to-br ${current.color} shadow-lg shadow-black/20`}>
                    {current.icon}
                  </div>
                  <div>
                    <h1 className="text-3xl font-bold text-white tracking-tight">{current.title}</h1>
                    <div className="flex items-center gap-2 mt-2 text-sm font-mono text-slate-400 bg-slate-950/50 px-3 py-1 rounded w-fit border border-slate-800 relative z-10">
                      <FileCode2 size={14} />
                      {current.id}
                    </div>
                  </div>
                </div>

                <div className="prose prose-invert max-w-none mb-10">
                  <p className="text-slate-300 text-lg leading-relaxed">
                    {current.description}
                  </p>
                </div>

                <h3 className="text-sm font-bold uppercase tracking-widest text-slate-500 mb-4 flex items-center gap-2">
                  <Info size={14} />
                  Core Capabilities
                </h3>

                <div className="grid gap-4 mt-auto">
                  {current.features.map((feature, i) => (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 + i * 0.05 }}
                      key={i}
                      className="bg-slate-950/40 border border-slate-800 rounded-xl p-4 flex items-start gap-4 transition-colors hover:bg-slate-950/60"
                    >
                      <div className={`mt-0.5 w-6 h-6 rounded-full bg-gradient-to-br flex items-center justify-center text-[10px] font-bold ${current.color}`}>
                        {i + 1}
                      </div>
                      <div className="text-slate-200 mt-0.5">
                        {feature}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

