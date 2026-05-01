export interface TelemetryPayload {
  tick: number;
  ts: string;
  global_state_norm: number;
  bus: {
    active_signals: number;
    mean_phase_coherence: number;
  };
  routing_matrix: number[][];
  lobes: {
    sensory: LobeStatus;
    memory: LobeStatus;
    semantic: LobeStatus;
    planning: LobeStatus;
    regulation: LobeStatus;
  };
  physics: {
    kappa: number;
    holonomy_trace: number;
    entropy: number;
  };
  asi_nodes: ASINode[];
}

export interface LobeStatus {
  norm: number;
  stable: boolean;
}

export interface ASINode {
  id: string;
  name: string;
  role: string;
  active: boolean;
  load: number;
  link_strength: number;
}
