# AstroLab 2.0 Roadmap: Towards Pure Architectures

This document outlines the development plan to transition AstroLab from a hybrid-baseline architecture to a system that demonstrates "Pure" logic for its respective engines (THRML and QNTM).

## 🚀 Vision
To differentiate the engines not just by name, but by **fundamental architecture**:
- **THRML:** Solves via **Thermodynamic Flow** on a Heterogeneous Graph. Constraints (Fuel, Smoothness) are structural nodes in the graph, not just post-processing checks.
- **QNTM:** Solves via **Quantum Exploration**. Visualizes the "Multiverse" of probabilities (superposition), tunneling through energy barriers to find optima that classical greedy methods miss.

---

## 📅 Phases

### Phase 1: The Heterogeneous Graph (THRML Engine)
**Goal:** Replace the simple 1D Ising chain with a complex constraint graph.

- [ ] **Data Structure Update (`engines/thrml/graph.py`)**
    - Implement a `HeterogeneousGraph` class.
    - Define node types: `TimeStepNode`, `FuelConstraintNode`, `ManeuverConstraintNode`.
- [ ] **Energy Function Logic**
    - Redefine energy $E$. Instead of just neighbors $(t, t+1)$, energy now includes global connections to the Fuel Node.
    - *Concept:* As more `TimeStepNodes` turn ON, the `FuelConstraintNode` "heats up", applying back-pressure (negative bias) to all nodes instantly.
- [ ] **Sampler Implementation**
    - Update Gibbs sampler to handle non-local connections (updating the Fuel Node updates effective fields for everyone).

### Phase 2: Quantum Visualization & Tunneling (QNTM Engine)
**Goal:** Simulate and visualize quantum mechanics behaviors.

- [ ] **Quantum State Representation (`engines/qntm/state.py`)**
    - Represent the solution not as a single path, but as a `WaveFunction` (probability distribution over paths).
- [ ] **Tunneling Simulation**
    - Implement a modified Simulated Annealing that allows "non-greedy" jumps (tunneling) through high-cost barriers.
    - Track "Ghost Trajectories" (rejected but explored paths) to visualize the quantum cloud.
- [ ] **Frontend Visualization (`frontend/.../QuantumView.tsx`)**
    - Create a "Tube of Probability" visualization instead of single lines.
    - Animate the "Collapse" of the wavefunction as the annealer cools down.

### Phase 3: Frontend Integration & Differentiation
**Goal:** Make the user *feel* the difference.

- [ ] **Engine Introspection UI**
    - **THRML Tab:** View the Graph Topology. See nodes "lighting up" and the Fuel Node exerting pressure.
    - **QNTM Tab:** View the Energy Landscape 3D plot and the wavefunction collapsing.
- [ ] **Performance Metrics**
    - Compare "Smoothness" (THRML specialty) vs "Global Optimality" (QNTM specialty) vs "Speed" (Random).

---

## 🛠 Next Immediate Steps
1. Open this folder (`AstroLab2.0`) as the new VS Code Workspace.
2. Initialize the `HeterogeneousGraph` class structure in `backend/engines/thrml`.
