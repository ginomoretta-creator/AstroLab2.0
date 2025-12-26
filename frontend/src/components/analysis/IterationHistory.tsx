import { useState, useMemo } from 'react'
import { useSimulationStore } from '../../stores/simulationStore'
import { Trophy, Target, Fuel, Clock, Gauge, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react'

const T_STAR_DAYS = 4.342 // 1 normalized time unit in days

type SortKey = 'iteration' | 'score' | 'dist' | 'fuel' | 'dv'
type SortDir = 'asc' | 'desc'

export default function IterationHistory() {
    const { iterationHistory, selectedResult, selectResult, params, status } = useSimulationStore()

    // Sort State
    const [sortKey, setSortKey] = useState<SortKey>('iteration')
    const [sortDir, setSortDir] = useState<SortDir>('asc')

    // Helper calculate Delta V (approx)
    const calculateDeltaVVal = (fraction: number | undefined) => {
        if (fraction === undefined) return 0
        const accel = params.thrust / params.mass
        const totalDurationSec = (params.numSteps * params.dt) * (T_STAR_DAYS * 24 * 3600)
        return accel * fraction * totalDurationSec
    }

    const calculateDays = () => {
        const totalDurationDays = (params.numSteps * params.dt) * T_STAR_DAYS
        return totalDurationDays.toFixed(1)
    }

    const sortedHistory = useMemo(() => {
        const items = [...iterationHistory]

        items.sort((a, b) => {
            let valA: number = 0
            let valB: number = 0

            switch (sortKey) {
                case 'iteration':
                    valA = a.iteration
                    valB = b.iteration
                    break
                case 'score':
                    valA = a.bestCost
                    valB = b.bestCost
                    break
                case 'dist':
                    valA = a.bestDistance || Infinity
                    valB = b.bestDistance || Infinity
                    break
                case 'fuel':
                    valA = a.bestThrustFraction || 0
                    valB = b.bestThrustFraction || 0
                    break
                case 'dv':
                    valA = calculateDeltaVVal(a.bestThrustFraction)
                    valB = calculateDeltaVVal(b.bestThrustFraction)
                    break
            }

            if (valA < valB) return sortDir === 'asc' ? -1 : 1
            if (valA > valB) return sortDir === 'asc' ? 1 : -1
            return 0
        })

        return items
    }, [iterationHistory, sortKey, sortDir, params])

    const SortIcon = ({ col }: { col: SortKey }) => {
        if (sortKey !== col) return <ArrowUpDown className="w-3 h-3 opacity-20" />
        return sortDir === 'asc' ? <ArrowUp className="w-3 h-3 text-[var(--accent-blue)]" /> : <ArrowDown className="w-3 h-3 text-[var(--accent-blue)]" />
    }

    const handleSort = (col: SortKey) => {
        if (sortKey === col) {
            setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
        } else {
            setSortKey(col)
            setSortDir('asc') // Default new column to asc usually? Or specific per col? Asc is fine.
        }
    }

    if (iterationHistory.length === 0) {
        return (
            <div className="h-full flex items-center justify-center text-theme-muted text-sm">
                {status === 'running'
                    ? 'Recording iteration history...'
                    : 'No history available. Run a simulation.'}
            </div>
        )
    }

    return (
        <div className="h-full flex flex-col">
            <div className="flex items-center justify-between mb-2 px-1">
                <h3 className="text-xs font-semibold text-theme-muted uppercase tracking-wider">
                    Iteration History
                </h3>
                <span className="text-xs text-theme-muted">
                    {iterationHistory.length} records
                </span>
            </div>

            <div className="flex-1 overflow-auto rounded-lg border border-theme bg-theme-primary/50 relative">
                <table className="w-full text-xs text-left">
                    <thead className="bg-theme-tertiary sticky top-0 z-10 text-theme-muted font-semibold select-none">
                        <tr>
                            <th
                                className="p-2 whitespace-nowrap cursor-pointer hover:bg-theme-elevated transition-colors"
                                onClick={() => handleSort('iteration')}
                            >
                                <div className="flex items-center gap-1">Iter <SortIcon col="iteration" /></div>
                            </th>
                            <th className="p-2 whitespace-nowrap">Method</th>
                            <th
                                className="p-2 whitespace-nowrap text-right cursor-pointer hover:bg-theme-elevated transition-colors"
                                onClick={() => handleSort('score')}
                            >
                                <div className="flex items-center justify-end gap-1">Score <SortIcon col="score" /></div>
                            </th>
                            <th
                                className="p-2 whitespace-nowrap text-right cursor-pointer hover:bg-theme-elevated transition-colors"
                                onClick={() => handleSort('dist')}
                            >
                                <div className="flex items-center justify-end gap-1">Dist <SortIcon col="dist" /></div>
                            </th>
                            <th
                                className="p-2 whitespace-nowrap text-right cursor-pointer hover:bg-theme-elevated transition-colors"
                                onClick={() => handleSort('fuel')}
                            >
                                <div className="flex items-center justify-end gap-1">Fuel <SortIcon col="fuel" /></div>
                            </th>
                            <th
                                className="p-2 whitespace-nowrap text-right cursor-pointer hover:bg-theme-elevated transition-colors"
                                onClick={() => handleSort('dv')}
                            >
                                <div className="flex items-center justify-end gap-1">Δv <SortIcon col="dv" /></div>
                            </th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-theme">
                        {sortedHistory.map((res) => {
                            const isSelected = selectedResult?.iteration === res.iteration
                            const deltaV = calculateDeltaVVal(res.bestThrustFraction).toFixed(1)

                            return (
                                <tr
                                    key={res.iteration}
                                    onClick={() => selectResult(res)}
                                    className={`
                                        cursor-pointer transition-colors hover:bg-theme-elevated
                                        ${isSelected ? 'bg-theme-secondary ring-1 ring-inset ring-[var(--accent-blue)]' : ''}
                                    `}
                                >
                                    <td className="p-2 font-mono text-theme-primary">
                                        #{res.iteration}
                                    </td>
                                    <td className="p-2 text-xs text-theme-secondary uppercase">
                                        {res.method || 'THRML'}
                                    </td>
                                    <td className="p-2 text-right font-mono text-theme-secondary">
                                        {res.bestCost.toFixed(4)}
                                    </td>
                                    <td className="p-2 text-right font-mono text-theme-muted">
                                        {res.bestDistance ? (res.bestDistance * 384400).toFixed(0) : '—'}
                                    </td>
                                    <td className="p-2 text-right font-mono text-theme-muted">
                                        {res.bestThrustFraction ? (res.bestThrustFraction * 100).toFixed(1) : '0'}%
                                    </td>
                                    <td className="p-2 text-right font-mono text-theme-muted">
                                        {deltaV}
                                    </td>
                                </tr>
                            )
                        })}
                    </tbody>
                </table>
            </div>

            <div className="mt-2 text-[10px] text-theme-muted px-1 flex gap-3">
                <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" /> Duration: {calculateDays()} days
                </span>
                <span className="flex items-center gap-1">
                    <Gauge className="w-3 h-3" /> Isp: {params.isp}s
                </span>
            </div>
        </div>
    )
}
