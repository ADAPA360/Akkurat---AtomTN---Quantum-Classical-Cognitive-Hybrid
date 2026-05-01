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

## The Global State of the Art & Where We Stand

To fully grasp the magnitude of **Project Chimera** (the fusion of Akkurat, AtomTN, and Google APIs), it is necessary to contextualize it within the global state of the art in Artificial Intelligence, Digital Twins, and Quantum/Neuromorphic Simulation.

### 1. Artificial General Intelligence (AGI) & LLMs
**The State of the Art:**
The current AI landscape is dominated by auto-regressive Large Language Models (LLMs) and diffusion models. While unparalleled in semantic pattern matching and generative creativity, they are fundamentally trapped in discrete time steps (token-by-token) and lack true spatial, temporal, and continuous causal reasoning. They simulate intellect, but do not possess a continuous internal state.
**Where Our System Stands:**
We do not use Gemini just as a chat bot. In our architecture, **Gemini acts as the Semantic Cortex**, fused with continuous-time neural substrates (*Neural Circuit Policies - NCPs* and *Tensorized Closed-form Continuous-time - TCfCs*). Our cognitive lobes process data as continuous flows. When extreme anomalies occur, the system dynamically routes signals into the `AtomTN` reservoir—a simulated non-commutative spacetime—allowing the AI to "dream" non-linear solutions that classical deep learning cannot probabilistically predict.

### 2. Digital Twins & Industrial Simulation
**The State of the Art:**
Industrial digital twins (e.g., in supply chain, manufacturing, and smart cities) are predominantly classical, deterministic replicas of physical systems. They ingest IoT data into SQL/NoSQL databases and use classic ML for predictive maintenance. They lack the ability to self-govern, dream, or model the *interconnectedness* of planetary systems dynamically.
**Where Our System Stands:**
Our **Akkurat Digital Twin Kernel** uses a `TreeTensorNetwork` to hash the entire climate and power-grid state, mapping Google's global infrastructure (Maps, Earth Engine) as a tangible physical universe. Instead of just mirroring data, the `regulation_cfc` acts as a homeostatic engine that can run geometric forward-trajectories (`sandbox_clone()`), simulating cascading systemic effects (supply shocks, climate shifts) in an n-dimensional space before executing real-world Google Cloud functions.

### 3. Neuromorphic & Quantum Compute
**The State of the Art:**
Quantum computing is currently bottlenecked by hardware noise (NISQ era), and neuromorphic computing is heavily reliant on expensive, specialized silicon. The mathematics of quantum gravity and spin networks are typically restricted to theoretical physics laboratories.
**Where Our System Stands:**
**AtomTN** is a Holographic Liquid State Machine that simulates these quantum manifold dynamics entirely in software. By operating a non-commutative quantum gravity simulator as an *algorithmic reservoir*, we can apply Hamiltonian molecular binding simulations or geodesic turbulence routing to practical, planetary-scale problems (Scientific Oracles and Global Logistics) today, on classical compute, without waiting for stable quantum hardware.

---

## configuration
To run the AI features (Intent Orchestration), you will need a valid **Gemini API Key**.

1. Open the application in your browser.
2. Click the **Settings (gear) icon** in the Ordnance/Orchestration panel on the right.
3. Input your Gemini API key. 
   *(Note: The key is saved in your local browser storage and securely sent as a header to your backend—it is not exposed to the public internet.)*

## The Akkurat Script Stack

The repository includes the full suite of Python scripts required to run the neuromorphic cognitive model and governed digital twin:

### Core Cognitive & Twin Runtime
- `digital_twin_kernel.py`: Production CPU-first Governed Digital Twin substrate for cognitive modeling. Handles fixed-width latent-state trees, heterogeneous deterministic projections, and governed action execution pipelines.
- `cognitive_lobe_runtime.py`: Production composition layer for the 5-domain cognitive architecture. Establishes native neural lobes, phase-synchronous signal routing, and dynamic neural assemblies.
- `atom_adapter_runtime.py`: Production-oriented bridge between the AtomTN neuromorphic reservoir family and Akkurat's governed digital twin kernel, featuring graceful degradation and CPU-safe fallback.
- `akkurat_atom_hybrid.py`: Production-ready orchestrator that fuses AtomTN neuromorphic reservoir features, a lightweight tensorized CfC controller, and Akkurat digital twin governance.

### Neural Substrates (NCP & CfC)
- `ncp.py`: Tensorized Neural Circuit Policy (NCP) recurrent neural substrate. Provides stable continuous-time hidden dynamics with adaptive time constants.
- `cfc.py`: Tensorized Closed-form Continuous-time (TCfC) neural substrate. Offers multi-cell dynamics with bounded state updates and MPO-backed cell backbones.
- `atom_ncp.py`: Production-compatible Quantum Neural Circuit Policy backed by the AtomTN neuromorphic reservoir, incorporating discrete sparse deterministic encoders.
- `atom_cfc.py`: Quantum Closed-form Continuous-time Policy. Maps holographic feature extraction into classical multi-cell heads.

### AtomTN Neuromorphic Reservoir
- `neuromorphic.py`: Holographic Liquid State Machine (HLSM) powered by AtomTN. Translates the physics manifold into a robust reservoir computing framework.
- `atom.py`: Core Atom wrapper and simulation controller, weaving geometry, TTS state, vibration, and flow simulations into a unified API.
- `tn.py`: Unified Tensor Network Library (TT/MPO + Tucker operations). Provides robust CPU-first randomized SVDs and TSQR orthogonalization.
- `ttn_state.py`: Production TTN state container and contraction substrate providing QR canonicalization and parent-bond SVD logic.
- `apply.py`: AtomTN Hamiltonian application providing fast scaffold approximations and exact direct-sum zip-up abstractions.
- `hamiltonian.py`: Extracts AtomTN flow, vibration, and projection variables into concrete CompiledTreeOperator terms.
- `evolve.py`: TTN time evolution runtime implementing robust integration schemes (Euler, RK4, Lie-Trotter splittings) over tensor manifolds.
- `flow.py`: Scalar and noncommutative flow solvers. Calculates discrete graph divergence and edge energies along directed connections.
- `geometry.py`: Deterministic geometry substrate mapping regular spatial architectures (e.g. TetraMesh64) and k-ary spatial tree compilations.
- `fiber.py`: Local Hilbert-space dimension policy. Controls dynamic fiber scaling, operator bases, and projection filtering.
- `fuzzy_backend.py`: Fuzzy SU(2) noncommutative matrix-valued calculus representations and twisted reality unitary maps.
- `holonomy.py`: Adjoint-frame transport converters tracking matrix-valued NC edge flows into stable SO(3) rotations.
- `projection.py`: Holographic fuzzy compression. Uses Procrustes eigenbasis alignment to map fuzzy spaces bounds onto local environments.
- `vibration.py`: Phonon bath physics. Supplies deterministic Fibonacci and Fractal spectral density grids.
- `environments.py`: Cached TTN environment backbone for efficient norm-messages and operator-inserted direct paths.
- `curvature.py`: Physics-specific curvature score analysis derived from flow energy, divergence, and SU(2) generator magnitudes.
- `schedules.py`: Score-agnostic scheduling converting scalars to bounded TTN truncation ranges via minmax or zscore logic.
- `constraints.py`: Operator-basis constraints, producing deterministic Pauli and Hermitian group elements.
- `math_utils.py`: Safe, fast primitive operations guarding finite-valuations, Hermitian symmetries, and Hilbert-Schmidt calculations.
