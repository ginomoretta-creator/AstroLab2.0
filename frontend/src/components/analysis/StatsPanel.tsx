import { useSimulationStore } from '../../stores/simulationStore'
import { Zap, Atom, Shuffle, Target, Gauge, Clock, TrendingDown } from 'lucide-react'

export default function StatsPanel() {
    const { status, currentMethod, currentIteration, params, results } = useSimulationStore()

    const currentResult = results[currentMethod]

    const methodIcons = {
        thrml: <Zap className="w-4 h-4" style={{ color: 'var(--accent-purple)' }} />,
        quantum: <Atom className="w-4 h-4" style={{ color: 'var(--accent-cyan)' }} />,
        random: <Shuffle className="w-4 h-4" style={{ color: 'var(--accent-orange)' }} />,
    }

    const statusColors = {
        idle: 'bg-theme-tertiary text-theme-muted',
        running: 'text-white',
        completed: 'text-white',
        error: 'text-white',
    }

    return (
        <div className="glass rounded-lg p-4 w-64 space-y-4 fade-in">
            {/* Status Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    {methodIcons[currentMethod]}
                    <span className="text-sm font-medium text-theme-primary capitalize">{currentMethod}</span>
                </div>
                <span
                    className={`text-xs px-2 py-0.5 rounded-full ${statusColors[status]}`}
                    style={{
                        backgroundColor: status === 'running' ? 'var(--accent-green)'
                            : status === 'completed' ? 'var(--accent-blue)'
                                : status === 'error' ? 'var(--accent-red)'
                                    : undefined
                    }}
                >
                    {status}
                </span>
            </div>

            {/* Progress */}
            {status === 'running' && (
                <div className="space-y-1">
                    <div className="flex justify-between text-xs text-theme-muted">
                        <span>Iteration</span>
                        <span className="font-mono">{currentIteration} / {params.numIterations}</span>
                    </div>
                    <div className="h-1.5 bg-theme-tertiary rounded-full overflow-hidden">
                        <div
                            className="h-full transition-all duration-300"
                            style={{
                                width: `${(currentIteration / params.numIterations) * 100}%`,
                                backgroundColor: 'var(--accent-blue)'
                            }}
                        />
                    </div>
                </div>
            )}

            {/* Stats */}
            <div className="space-y-2">
                <StatRow
                    icon={<Target className="w-3.5 h-3.5" />}
                    label="Obj. Score"
                    value={currentResult?.bestCost?.toFixed(4) || '—'}
                />
                <StatRow
                    icon={<Clock className="w-3.5 h-3.5" />}
                    label="Samples/Iter"
                    value={params.batchSize.toString()}
                />
            </div>

            {/* Comparison Summary */}
            {(results.thrml || results.quantum || results.random) && (
                <div className="pt-3 border-t border-theme space-y-2">
                    <div className="flex items-center gap-1.5 text-xs font-medium text-theme-muted mb-2">
                        <TrendingDown className="w-3.5 h-3.5" />
                        Method Comparison
                    </div>
                    {results.thrml && (
                        <CompareRow
                            method="THRML"
                            cost={results.thrml.bestCost}
                            colorVar="--accent-purple"
                            isBest={findBestMethod() === 'thrml'}
                        />
                    )}
                    {results.quantum && (
                        <CompareRow
                            method="Quantum"
                            cost={results.quantum.bestCost}
                            colorVar="--accent-cyan"
                            isBest={findBestMethod() === 'quantum'}
                        />
                    )}
                    {results.random && (
                        <CompareRow
                            method="Random"
                            cost={results.random.bestCost}
                            colorVar="--accent-orange"
                            isBest={findBestMethod() === 'random'}
                        />
                    )}
                </div>
            )}
        </div>
    )

    function findBestMethod() {
        const methods = ['thrml', 'quantum', 'random'] as const
        let best = null
        let bestCost = Infinity

        for (const m of methods) {
            if (results[m] && results[m]!.bestCost < bestCost) {
                bestCost = results[m]!.bestCost
                best = m
            }
        }
        return best
    }
}

interface StatRowProps {
    icon: React.ReactNode
    label: string
    value: string
}

function StatRow({ icon, label, value }: StatRowProps) {
    return (
        <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-theme-muted">
                {icon}
                <span className="text-xs">{label}</span>
            </div>
            <span className="text-sm font-mono text-theme-primary">{value}</span>
        </div>
    )
}

interface CompareRowProps {
    method: string
    cost: number
    colorVar: string
    isBest: boolean
}

function CompareRow({ method, cost, colorVar, isBest }: CompareRowProps) {
    return (
        <div className={`flex items-center justify-between text-xs ${isBest ? 'font-semibold' : ''}`}>
            <div className="flex items-center gap-2">
                <div
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: `var(${colorVar})` }}
                ></div>
                <span className="text-theme-secondary">{method}</span>
                {isBest && (
                    <span
                        className="text-[10px] px-1 rounded"
                        style={{
                            backgroundColor: `color-mix(in srgb, var(${colorVar}) 20%, transparent)`,
                            color: `var(${colorVar})`
                        }}
                    >
                        BEST
                    </span>
                )}
            </div>
            <span className="font-mono text-theme-muted">{cost.toFixed(4)}</span>
        </div>
    )
}
