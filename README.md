# AstroLab 2.0

AstroLab 2.0 is a cutting-edge Cislunar Trajectory Sandbox that leverages **Quantum-Assisted** and **Probabilistic Warm-Starting** techniques for low-thrust trajectory optimization. It combines a high-performance Python backend (powered by JAX) with a modern React/Three.js frontend to visualize complex orbital mechanics in real-time.

## Features

- **Trajectory Optimization**: Uses thermal (probabilistic) and quantum-inspired engines to find optimal paths to the Moon.
- **Physics Engine**: Accurate Cislunar physics simulation using restricted 3-body problem dynamics.
- **3D Visualization**: Interactive 3D view of Earth, Moon, and satellite trajectories.
- **Real-time Tweaking**: Adjust mass, thrust, and simulation parameters on the fly.

## Prerequisites

- **Node.js**: v18 or higher (for the frontend).
- **Python**: v3.9 or higher (for the backend).

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ginomoretta-creator/AstroLab2.0.git
    cd AstroLab2.0
    ```

### Backend Setup

The backend handles the physics and optimization engine.

1.  Navigate to the `backend` directory (or root if running from there, but ensuring python path is correct):
    ```bash
    cd backend
    ```
    *(Note: You can run it from the root directory as well, provided your PYTHONPATH is set correctly, but running from `backend` is adhering to typical structures).*

2.  (Optional but recommended) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install fastapi uvicorn pydantic numpy jax jaxlib
    ```
    *(Note: `jax` installation might vary based on your system, especially for GPU support. See [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for details).*

4.  **Run the Server:**
    ```bash
    python server.py
    ```
    The server will start on `http://localhost:8080`.

### Frontend Setup

The frontend provides the user interface.

1.  Open a new terminal and navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```

2.  Install dependencies:
    ```bash
    npm install
    # or if you use yarn
    yarn install
    ```

3.  **Run the Application:**
    ```bash
    npm run dev
    ```
    The application will typically start on `http://localhost:5173`. Open this link in your browser.

## Usage

1.  Ensure the **Backend** is running (`python server.py`).
2.  Open the **Frontend** in your browser.
3.  Use the control panel on the left to:
    *   Select the optimization method (`THRML` or `Quantum`).
    *   Adjust satellite parameters (Mass, Thrust, Altitude).
    *   Click **"Start Simulation"**.
4.  Watch as the trajectories are generated and optimized in real-time!

## Troubleshooting

- **Backend Connection Error**: Ensure the backend server is running on port 8080. Check the console logs in the frontend for connection issues.
- **Performance**: Large simulation steps or high batch sizes may slow down your machine. Adjust `num_steps` or `batch_size` in the code (`server.py`) if necessary.
