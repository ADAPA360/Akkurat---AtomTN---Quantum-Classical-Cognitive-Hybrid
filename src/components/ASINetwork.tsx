import { motion } from "motion/react";
import { useEffect, useState, useMemo } from "react";
import { ASINode } from "../types";

const ASI_COLORS: Record<string, string> = {
  astra: "#3b82f6", // blue
  notebooklm: "#10b981", // emerald
  gemma: "#8b5cf6", // violet
  imagefx: "#f43f5e", // purple
  musicfx: "#f59e0b", // amber
  textfx: "#06b6d4", // cyan
  alphafold: "#ec4899", // rose
  alphageometry: "#6366f1", // purple stronger
  illuminate: "#14b8a6", // teal
};

export default function ASINetwork({ nodes, drift }: { nodes: ASINode[], drift: number }) {
  const [rotated, setRotated] = useState(0);

  // Slow continuous rotation of the parent canvas
  useEffect(() => {
    let animationFrame: number;
    const animate = () => {
      setRotated((r) => (r + 0.1) % 360);
      animationFrame = requestAnimationFrame(animate);
    };
    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, []);

  const centerX = 300;
  const centerY = 300;
  const AGI_RADIUS = 60 + drift * 20; 
  const ORBIT_RADIUS = 180 + drift * 50;

  return (
    <div className="relative w-full h-full flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-radial from-slate-900 via-transparent to-transparent opacity-50 point-events-none" />
      
      <svg width="600" height="600" className="overflow-visible filter drop-shadow-[0_0_15px_rgba(0,0,0,0.5)]">
        <defs>
          <radialGradient id="coreGlow">
            <stop offset="0%" stopColor="#c084fc" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#4c1d95" stopOpacity="0.1" />
          </radialGradient>
        </defs>

        <g style={{ transformOrigin: '300px 300px', transform: `rotate(${rotated}deg)` }}>
          {/* Orbit rings */}
          <circle cx={centerX} cy={centerY} r={ORBIT_RADIUS} className="stroke-slate-800" fill="none" strokeWidth="1" strokeDasharray="4 4" />
          <circle cx={centerX} cy={centerY} r={ORBIT_RADIUS * 1.3} className="stroke-slate-800/50" fill="none" strokeWidth="1" strokeDasharray="8 8" />

          {/* Links from Core to ASI Nodes */}
          {nodes.map((node, i) => {
            const angle = (i / nodes.length) * 2 * Math.PI;
            const targetX = centerX + Math.cos(angle) * ORBIT_RADIUS;
            const targetY = centerY + Math.sin(angle) * ORBIT_RADIUS;
            const tension = Math.max(0.1, node.link_strength);
            const color = ASI_COLORS[node.id] || "#fff";

            // Bezier curve to simulate topological fiber bundles
            const cpX = centerX + Math.cos(angle) * (ORBIT_RADIUS * 0.5) - Math.sin(angle) * 50;
            const cpY = centerY + Math.sin(angle) * (ORBIT_RADIUS * 0.5) + Math.cos(angle) * 50;

            return (
              <motion.path
                key={`link-${node.id}`}
                d={`M ${centerX} ${centerY} Q ${cpX} ${cpY} ${targetX} ${targetY}`}
                fill="none"
                stroke={color}
                strokeWidth={node.load * 4}
                className="opacity-60 mix-blend-screen"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1, strokeOpacity: [0.3, node.load, 0.3] }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              />
            );
          })}

          {/* ASI Nodes */}
          {nodes.map((node, i) => {
            const angle = (i / nodes.length) * 2 * Math.PI;
            const x = centerX + Math.cos(angle) * ORBIT_RADIUS;
            const y = centerY + Math.sin(angle) * ORBIT_RADIUS;
            const color = ASI_COLORS[node.id] || "#fff";

            return (
              <g key={`node-${node.id}`}>
                <motion.circle
                  cx={x}
                  cy={y}
                  r={8 + node.load * 12}
                  fill={color}
                  className="mix-blend-screen"
                  animate={{ scale: [1, 1 + node.load * 0.5, 1], filter: [`drop-shadow(0 0 5px ${color})`, `drop-shadow(0 0 15px ${color})`, `drop-shadow(0 0 5px ${color})`] }}
                  transition={{ duration: 1.5 + Math.random(), repeat: Infinity }}
                />
                
                {/* Counter-rotation for text to stay readable */}
                <g style={{ transformOrigin: `${x}px ${y}px`, transform: `rotate(${-rotated}deg)` }}>
                  <rect x={x + 15} y={y - 20} width={node.name.length * 7 + 10} height="20" rx="4" fill="rgba(15,23,42,0.8)" className="stroke-slate-700 stroke-[1px]" />
                  <text
                    x={x + 20}
                    y={y - 6}
                    fill="#e2e8f0"
                    className="text-[10px] font-mono tracking-wider font-semibold pointer-events-none select-none"
                  >
                    {node.name.toUpperCase()}
                  </text>
                  <text
                    x={x + 20}
                    y={y + 10}
                    fill="#94a3b8"
                    className="text-[8px] font-sans tracking-wide pointer-events-none select-none"
                  >
                    Load: {(node.load * 100).toFixed(0)}%
                  </text>
                </g>
              </g>
            );
          })}

          {/* Central AGI Core (AtomTN) */}
          <motion.circle
            cx={centerX}
            cy={centerY}
            r={AGI_RADIUS}
            fill="url(#coreGlow)"
            className="stroke-purple-500/50"
            strokeWidth="2"
            animate={{
              r: [AGI_RADIUS, AGI_RADIUS * 1.1, AGI_RADIUS],
              strokeWidth: [2, 6, 2],
            }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          />

          <g style={{ transformOrigin: `${centerX}px ${centerY}px`, transform: `rotate(${-rotated}deg)` }}>
             <text
                x={centerX}
                y={centerY}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="#f8fafc"
                className="text-xs font-mono font-bold tracking-[0.2em] pointer-events-none select-none"
              >
                AKKURAT
              </text>
              <text
                x={centerX}
                y={centerY + 14}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="#cbd5e1"
                className="text-[8px] font-mono tracking-wider pointer-events-none select-none"
              >
                TETRAMESH64 CORE
              </text>
          </g>
        </g>
      </svg>
    </div>
  );
}
