import { useMemo } from 'react'
import { useSimulationStore } from '../../stores/simulationStore'

export default function ScheduleVisualization() {
    const { results, currentMethod, params, status, selectedResult } = useSimulationStore()

    const displayResult = selectedResult || results[currentMethod]
    const method = selectedResult?.method || currentMethod

    // Get schedule from best trajectory (approximate thrust pattern)
    const schedule = useMemo(() => {
        // Prefer backend-provided schedule if available
        if (displayResult?.bestSchedule && displayResult.bestSchedule.length > 0) {
            // Backend provides ternary schedule (-1, 0, 1) directly
            return displayResult.bestSchedule
        }

        if (!displayResult?.bestTrajectory || displayResult.bestTrajectory.length < 2) {
            return []
        }

        // Compute velocity changes to infer thrust
        // Fallback robust logic
        const traj = displayResult.bestTrajectory
        const computedSchedule: number[] = []

        // Analyze points. If we have mass (5th state), use mass depletion
        // If not, we can't easily infer thrust from position/velocity without more math.
        // But usually backend provides schedule.
        // If we are here, it means backend failed to provide schedule.

        // Downsample check
        const step = Math.max(1, Math.floor(traj.length / 500))

        for (let i = step; i < traj.length; i += step) {
            if (traj[i].length >= 5) {
                // Check mass decrease
                const dm = traj[i - step][4] - traj[i][4]
                // dm > 0 implies thrust (since mdot = -T/Isp)
                computedSchedule.push(dm > 1e-9 ? 1 : 0)
            } else {
                computedSchedule.push(0)
            }
        }
        return computedSchedule
    }, [displayResult])

    const thrustCount = schedule.filter(s => s === 1).length
    const thrustFraction = schedule.length > 0
        ? (thrustCount / schedule.length) * 100
        : displayResult?.bestThrustFraction !== undefined
            ? displayResult.bestThrustFraction * 100
            : 0

    // Downsample for display if too many steps
    const displaySchedule = useMemo(() => {
        if (schedule.length <= 200) return schedule
        const chunkSize = Math.ceil(schedule.length / 200)
        const downsampled = []
        for (let i = 0; i < schedule.length; i += chunkSize) {
            const chunk = schedule.slice(i, i + chunkSize)
            const sum = chunk.reduce((a, b) => a + b, 0)
            downsampled.push(sum / chunk.length) // 0 to 1 value
        }
        return downsampled
    }, [schedule])

    if (!displayResult) {
        return (
            <div className="h-full flex items-center justify-center text-theme-muted text-xs">
                Waiting for results...
            </div>
        )
    }

    const getColor = () => {
        switch (method) {
            case 'thrml': return 'bg-[var(--accent-purple)]'
            case 'quantum': return 'bg-[var(--accent-cyan)]'
            case 'random': return 'bg-[var(--accent-orange)]'
            default: return 'bg-theme-secondary'
        }
    }

    return (
        <div className="h-full flex flex-col p-2">
            <div className="flex justify-between items-center mb-1">
                <span className="text-xs font-semibold text-theme-muted uppercase tracking-wider">Thrust Schedule</span>
                <span className="text-xs text-theme-muted font-mono">{thrustFraction.toFixed(1)}% Usage</span>
            </div>

            <div className="flex-1 flex items-end gap-[1px] h-full bg-theme-elevated/30 rounded p-1 overflow-hidden">
                {displaySchedule.map((val, i) => {
                    const absVal = Math.abs(val)
                    const isBrake = val < -0.1
                    const isThrust = val > 0.1

                    return (
                        <div
                            key={i}
                            className={`flex-1 rounded-sm transition-all duration-300 ${isBrake ? 'bg-red-500' : isThrust ? getColor() : 'bg-transparent'}`}
                            style={{
                                height: (isThrust || isBrake) ? `${Math.max(20, absVal * 100)}%` : '2px', // Minimum visibility
                                opacity: (isThrust || isBrake) ? 0.8 + (absVal * 0.2) : 0.1
                            }}
                            title={`Thrust: ${(val * 100).toFixed(0)}%`}
                        />
                    )
                })}
            </div>

            <div className="flex justify-between text-[10px] text-theme-muted mt-1 px-1">
                <span>Start</span>
                <span>{((displayResult.totalIterations || 0) > 0 ? `Iter #${displayResult.iteration}` : 'Live')}</span>
                <span>End</span>
            </div>
        </div>
    )
}
