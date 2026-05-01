# Akkurat / AtomTN Digital Twin

A cognitive architecture digital twin demonstrating quantum manifold orchestration and ASI network spawning. This application uses the Gemini 3.1 Pro model to process continuous intelligence and route intents to specific simulated Labs nodes.

## Features
- **Cognitive Cortex**: Real-time visualization of sensory, semantic, memory, and planning telemetry.
- **Planetary Symbiosis Graph**: Node-based geometric flow visualization of ASI execution manifolds.
- **Natural Language Orchestration**: Inject intents to trigger dynamic LLM-driven orchestration.

## Setup & Run Locally

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
4. Access the application at `http://localhost:3000`.

## Configuration
To run the AI features (Intent Orchestration), you will need a valid **Gemini API Key**.

1. Open the application in your browser.
2. Click the **Settings (gear) icon** in the Ordnance/Orchestration panel on the right.
3. Input your Gemini API key. 
   *(Note: The key is saved in your local browser storage and securely sent as a header to your backend—it is not exposed to the public internet.)*

## The Akkurat Script Stack

The repository includes the full suite of Python scripts required to run the neuromorphic cognitive model and governed digital twin:

### Core Cognitive & Twin Runtime
- `digital_twin_kernel.py`: Governed digital twin kernel and tree tensor network backend.
- `cognitive_lobe_runtime.py`: Production composition layer for the 5-domain cognitive runtime.
- `atom_adapter_runtime.py`: Bridge between the AtomTN neuromorphic reservoir and Akkurat's governed digital twin kernel.
- `akkurat_atom_hybrid.py`: Hybrid orchestrator that fuses AtomTN reservoir features with a CfC controller and Akkurat twin.

### Neural Substrates (NCP & CfC)
- `ncp.py`: Tensorized Neural Circuit Policy (NCP) recurrent neural substrate.
- `cfc.py`: Tensorized Closed-form Continuous-time (TCfC) neural substrate.
- `atom_ncp.py`: Quantum Neural Circuit Policy backed by the AtomTN neuromorphic reservoir.
- `atom_cfc.py`: Quantum Closed-form Continuous-time Policy backed by the AtomTN neuromorphic reservoir.

### AtomTN Neuromorphic Reservoir
- `neuromorphic.py`: Holographic Liquid State Machine (HLSM) powered by AtomTN.
- `atom.py`: Core Atom wrapper and simulation controller.
- `tn.py`: Unified Tensor Network Library (TT/MPO + Tucker operations).
- `ttn_state.py`: Tree Tensor Network (TTN) state container and generic tensor contraction operations.
- `flow.py`: Scalar and noncommutative flow solvers and diagnostics.
- `hamiltonian.py`: Hamiltonian building rules and observables setup.
- `evolve.py`: TTN time evolution runtime (legacy and RK/Lie-Trotter step methods).
- `geometry.py`: Spatial graph and k-ary spatial tree (e.g., TetraMesh64) compilation.
- `apply.py`: Quantum operator application logic and direct sum zip-up layer.
- `fiber.py`: Local fiber scheduling and basis construction for leaves.
- `fuzzy_backend.py`: Fuzzy SU(2) and non-commutative matrix-valued calculus representations.
- `holonomy.py`: Adjoint-frame transport for non-commutative flow edges.
- `projection.py`: Fuzzy k -> local d holographic compression and projections.
- `vibration.py`: Phonon bath physics and vibration grid models.
- `environments.py`: Cached TTN environments and bottom-up/top-down message passing.
- `curvature.py`: Curvature and flow-derived score utilities.
- `schedules.py`: Generic scheduling utilities for score and integer parameter mappings.
- `constraints.py`: Operator-basis constraints/hooks (e.g., AdinkraConstraint).
- `math_utils.py`: Safe, fast primitive math operations and dense tensor algebra helpers.
