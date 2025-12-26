import { useState } from 'react'
import {
    Play,
    Pause,
    RotateCcw,
    Zap,
    Atom,
    Shuffle,
    ChevronDown,
    ChevronRight,
    Settings,
    BarChart3,
    GitCompare
} from 'lucide-react'
import { useSimulationStore, SimulationMethod } from '../../stores/simulationStore'
import { useBackend } from '../../hooks/useBackend'
import ParameterSlider from '../simulation/ParameterSlider'

const methodConfig: Record<SimulationMethod, { icon: React.ReactNode; label: string; colorVar: string }> = {
    thrml: {
        icon: <Zap className="w-4 h-4" />,
        label: 'THRML (Probabilistic)',
        colorVar: '--accent-purple'
    },
    quantum: {
        icon: <Atom className="w-4 h-4" />,
        label: 'Quantum Annealing',
        colorVar: '--accent-cyan'
    },
    random: {
        icon: <Shuffle className="w-4 h-4" />,
        label: 'Random Baseline',
        colorVar: '--accent-orange'
    },
}

export default function Sidebar() {
    const {
        status,
        currentMethod,
        params,
        backendStatus,
        setMethod,
        updateParams,
        clearResults,
        setBackendStatus,
    } = useSimulationStore()

    const { startSimulation, stopSimulation } = useBackend()

    const [paramsExpanded, setParamsExpanded] = useState(true)
    const [advancedExpanded, setAdvancedExpanded] = useState(false)
    const [isRestarting, setIsRestarting] = useState(false)

    const handleSimulation = async () => {
        if (status === 'running') {
            stopSimulation()
        } else {
            await startSimulation()
        }
    }

    const isBackendReady = backendStatus === 'online'

    const restartBackend = async () => {
        if (!window.electronAPI?.restartBackend) return
        setIsRestarting(true)
        setBackendStatus('checking')
        try {
            await window.electronAPI.restartBackend()
            // Give the process a moment to come back up
            setTimeout(() => setBackendStatus('online'), 2000)
        } catch (e) {
            setBackendStatus('offline')
        } finally {
            setIsRestarting(false)
        }
    }

    return (
        <aside className="w-72 bg-theme-secondary border-r border-theme flex flex-col overflow-hidden">
            {/* Method Selection */}
            <div className="p-4 border-b border-theme">
                <h3 className="text-xs font-semibold text-theme-muted uppercase tracking-wider mb-3">
                    Generation Method
                </h3>
                <div className="space-y-2">
                    {(['thrml', 'quantum', 'random'] as SimulationMethod[]).map((method) => {
                        const config = methodConfig[method]
                        const isSelected = currentMethod === method

                        return (
                            <button
                                key={method}
                                onClick={() => setMethod(method)}
                                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${isSelected
                                    ? 'border'
                                    : 'bg-theme-tertiary hover:bg-theme-elevated border border-transparent'
                                    }`}
                                style={isSelected ? {
                                    backgroundColor: `color-mix(in srgb, var(${config.colorVar}) 20%, transparent)`,
                                    borderColor: `color-mix(in srgb, var(${config.colorVar}) 50%, transparent)`,
                                } : {}}
                            >
                                <div style={{ color: `var(${config.colorVar})` }}>
                                    {config.icon}
                                </div>
                                <span className="text-sm font-medium text-theme-primary">{config.label}</span>
                            </button>
                        )
                    })}
                </div>
            </div>

            {/* Parameters */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {/* Basic Parameters */}
                <div>
                    <button
                        onClick={() => setParamsExpanded(!paramsExpanded)}
                        className="flex items-center gap-2 text-xs font-semibold text-theme-muted uppercase tracking-wider mb-3 hover:text-theme-primary transition-colors"
                    >
                        {paramsExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                        <Settings className="w-3.5 h-3.5" />
                        Parameters
                    </button>
                    {paramsExpanded && (
                        <div className="space-y-4 fade-in">
                            <ParameterSlider
                                label="Time Steps"
                                value={params.numSteps}
                                min={100}
                                max={120000}
                                step={100}
                                onChange={(v) => updateParams({ numSteps: v })}
                            />
                            <ParameterSlider
                                label="Batch Size"
                                value={params.batchSize}
                                min={10}
                                max={200}
                                step={10}
                                onChange={(v) => updateParams({ batchSize: v })}
                            />
                            <ParameterSlider
                                label="Coupling Strength"
                                value={params.couplingStrength}
                                min={0.1}
                                max={5}
                                step={0.1}
                                onChange={(v) => updateParams({ couplingStrength: v })}
                            />
                            <ParameterSlider
                                label="Spacecraft Mass (kg)"
                                value={params.mass}
                                min={100}
                                max={5000}
                                step={100}
                                onChange={(v) => updateParams({ mass: v })}
                            />
                            <ParameterSlider
                                label="Thrust (N)"
                                value={params.thrust}
                                min={0.1}
                                max={5}
                                step={0.1}
                                onChange={(v) => updateParams({ thrust: v })}
                            />
                            <ParameterSlider
                                label="Isp (s)"
                                value={params.isp}
                                min={1000}
                                max={5000}
                                step={100}
                                onChange={(v) => updateParams({ isp: v })}
                            />
                        </div>
                    )}
                </div>

                {/* Advanced Parameters */}
                <div>
                    <button
                        onClick={() => setAdvancedExpanded(!advancedExpanded)}
                        className="flex items-center gap-2 text-xs font-semibold text-theme-muted uppercase tracking-wider mb-3 hover:text-theme-primary transition-colors"
                    >
                        {advancedExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                        <BarChart3 className="w-3.5 h-3.5" />
                        Advanced
                    </button>
                    {advancedExpanded && (
                        <div className="space-y-4 fade-in">
                            <ParameterSlider
                                label="Initial Altitude (km)"
                                value={params.initialAltitude}
                                min={200}
                                max={2000}
                                step={50}
                                onChange={(v) => updateParams({ initialAltitude: v })}
                            />
                            <ParameterSlider
                                label="Time Step (dt)"
                                value={params.dt}
                                min={0.0001}
                                max={0.01}
                                step={0.0001}
                                onChange={(v) => updateParams({ dt: v })}
                            />
                            <ParameterSlider
                                label="Iterations"
                                value={params.numIterations}
                                min={5}
                                max={100}
                                step={5}
                                onChange={(v) => updateParams({ numIterations: v })}
                            />
                        </div>
                    )}
                </div>
            </div>

            {/* Action Buttons */}
            <div className="p-4 border-t border-theme space-y-2">
                {/* Main Simulation Button */}
                <button
                    onClick={handleSimulation}
                    disabled={!isBackendReady && status !== 'running'}
                    className={`w-full flex items-center justify-center gap-2 py-3 rounded-lg font-semibold transition-all ${!isBackendReady
                        ? 'bg-theme-tertiary text-theme-muted cursor-not-allowed'
                        : status === 'running'
                            ? 'text-white'
                            : 'text-white hover:brightness-110'
                        }`}
                    style={{
                        backgroundColor: !isBackendReady
                            ? undefined
                            : status === 'running'
                                ? 'var(--accent-red)'
                                : 'var(--accent-blue)'
                    }}
                >
                    {status === 'running' ? (
                        <>
                            <Pause className="w-5 h-5" />
                            Stop Simulation
                        </>
                    ) : (
                        <>
                            <Play className="w-5 h-5" />
                            {isBackendReady ? 'Start Simulation' : 'Backend Offline'}
                        </>
                    )}
                </button>

                {/* Compare All Methods */}
                <button
                    onClick={() => {
                        // TODO: Run all methods and compare
                    }}
                    disabled={!isBackendReady || status === 'running'}
                    className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-theme-tertiary hover:bg-theme-elevated text-theme-primary transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <GitCompare className="w-4 h-4" />
                    Compare All Methods
                </button>

                {/* Clear Results */}
                <button
                    onClick={clearResults}
                    className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-theme-tertiary hover:bg-theme-elevated text-theme-secondary transition-all"
                >
                    <RotateCcw className="w-4 h-4" />
                    Clear Results
                </button>

                {/* Restart Backend */}
                <button
                    onClick={restartBackend}
                    disabled={isRestarting}
                    className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-theme-tertiary hover:bg-theme-elevated text-theme-secondary transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <RotateCcw className={`w-4 h-4 ${isRestarting ? 'animate-spin' : ''}`} />
                    {isRestarting ? 'Restarting Backend...' : 'Restart Backend'}
                </button>
            </div>
        </aside>
    )
}
