import { create } from 'zustand'

export type SimulationMethod = 'thrml' | 'quantum' | 'random'
export type SimulationStatus = 'idle' | 'running' | 'completed' | 'error'

export interface Trajectory {
    points: [number, number][]
    cost: number
    method: SimulationMethod
}

export interface SimulationParams {
    numSteps: number
    batchSize: number
    couplingStrength: number
    mass: number
    thrust: number
    isp: number
    initialAltitude: number
    dt: number
    numIterations: number
}

export interface SimulationResult {
    iteration: number
    totalIterations: number
    trajectories: number[][][]
    bestTrajectory: number[][]
    bestCost: number
    bestSchedule?: number[]
    bestThrustFraction?: number
    bestDistance?: number
    method?: SimulationMethod
}

interface SimulationState {
    // Status
    status: SimulationStatus
    currentMethod: SimulationMethod
    backendStatus: 'checking' | 'online' | 'offline'
    backendPort: number
    errorMessage: string | null

    // Parameters
    params: SimulationParams

    // Results
    currentIteration: number
    results: {
        thrml: SimulationResult | null
        quantum: SimulationResult | null
        random: SimulationResult | null
    }

    // History
    trajectoryHistory: Trajectory[]
    iterationHistory: SimulationResult[]
    selectedResult: SimulationResult | null

    // Actions
    setStatus: (status: SimulationStatus) => void
    setMethod: (method: SimulationMethod) => void
    setBackendStatus: (status: 'checking' | 'online' | 'offline') => void
    setBackendPort: (port: number) => void
    setError: (message: string | null) => void
    updateParams: (params: Partial<SimulationParams>) => void
    addResult: (method: SimulationMethod, result: SimulationResult) => void
    addTrajectory: (trajectory: Trajectory) => void
    selectResult: (result: SimulationResult | null) => void
    clearResults: () => void
    reset: () => void
}

const defaultParams: SimulationParams = {
    numSteps: 120000,      // ~250 days with dt=0.0005
    batchSize: 100,        // Increased for better exploration
    couplingStrength: 1.0,
    mass: 500,             // Smaller sat for realistic low-thrust
    thrust: 0.5,           // Typical Hall thruster
    isp: 3000,             // Hall thruster Isp
    initialAltitude: 200,  // LEO parking orbit
    dt: 0.0005,            // Very small steps for smooth trajectories
    numIterations: 30,     // More iterations for convergence
}

export const useSimulationStore = create<SimulationState>((set) => ({
    // Initial state
    status: 'idle',
    currentMethod: 'thrml',
    backendStatus: 'checking',
    backendPort: 8080,
    errorMessage: null,
    params: defaultParams,
    currentIteration: 0,
    results: {
        thrml: null,
        quantum: null,
        random: null,
    },
    trajectoryHistory: [],

    iterationHistory: [],
    selectedResult: null,

    // Actions
    setStatus: (status) => set({ status }),
    setMethod: (method) => set({ currentMethod: method }),
    setBackendStatus: (status) => set({ backendStatus: status }),
    setBackendPort: (port) => set({ backendPort: port }),
    setError: (message) => set({ errorMessage: message }),

    updateParams: (newParams) => set((state) => ({
        params: { ...state.params, ...newParams }
    })),

    addResult: (method, result) => set((state) => {
        // Add to full iteration history if it's a new iteration
        // We use the iteration number to avoid duplicates
        const existingIdx = state.iterationHistory.findIndex(r => r.iteration === result.iteration)
        let newIterationHistory = [...state.iterationHistory]

        if (existingIdx >= 0) {
            // Update existing
            newIterationHistory[existingIdx] = result
        } else {
            newIterationHistory.push(result)
        }
        // Sort by iteration
        newIterationHistory.sort((a, b) => a.iteration - b.iteration)

        return {
            results: { ...state.results, [method]: result },
            currentIteration: result.iteration,
            iterationHistory: newIterationHistory
        }
    }),

    selectResult: (result) => set({ selectedResult: result }),

    addTrajectory: (trajectory) => set((state) => ({
        trajectoryHistory: [...state.trajectoryHistory.slice(-50), trajectory]
    })),

    clearResults: () => set({
        results: { thrml: null, quantum: null, random: null },
        trajectoryHistory: [],
        currentIteration: 0,
        errorMessage: null,
    }),

    reset: () => set({
        status: 'idle',
        params: defaultParams,
        currentIteration: 0,
        results: { thrml: null, quantum: null, random: null },
        trajectoryHistory: [],
        errorMessage: null,
    }),
}))
