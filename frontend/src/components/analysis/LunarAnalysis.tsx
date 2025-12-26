import { useRef, useEffect, useState } from 'react'
import { useSimulationStore } from '../../stores/simulationStore'
import { useThemeStore } from '../../stores/themeStore'
import { ZoomIn, ZoomOut, Move } from 'lucide-react'

// Constants
const MU = 0.01215058560962404
const L_STAR = 384400 // km
const R_MOON_KM = 1737.4
const R_MOON_NORM = R_MOON_KM / L_STAR
const BASE_VIEW_RADIUS_NORM = 50000 / L_STAR

export default function LunarAnalysis() {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const { results, currentMethod, selectedResult } = useSimulationStore()
    const { theme } = useThemeStore()

    // Interactive State
    const [zoom, setZoom] = useState(1.0)
    const [pan, setPan] = useState({ x: 0, y: 0 })
    const [isDragging, setIsDragging] = useState(false)
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

    const displayResult = selectedResult || results[currentMethod]
    const displayMethod = selectedResult?.method || currentMethod

    // Interaction Handlers
    const handleWheel = (e: React.WheelEvent) => {
        // Zoom towards mouse pointer could be managed here, but center zoom is simpler for now
        const delta = -Math.sign(e.deltaY) * 0.1
        setZoom(z => Math.max(0.1, Math.min(10, z + delta)))
    }

    const handleMouseDown = (e: React.MouseEvent) => {
        setIsDragging(true)
        setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
    }

    const handleMouseMove = (e: React.MouseEvent) => {
        if (isDragging) {
            setPan({
                x: e.clientX - dragStart.x,
                y: e.clientY - dragStart.y
            })
        }
    }

    const handleMouseUp = () => {
        setIsDragging(false)
    }

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Handle resizing
        const resizeHelper = () => {
            const parent = canvas.parentElement
            if (parent) {
                canvas.width = parent.clientWidth
                canvas.height = parent.clientHeight
            }
        }
        resizeHelper()
        window.addEventListener('resize', resizeHelper)

        // Rendering Loop
        const render = () => {
            const width = canvas.width
            const height = canvas.height
            const cx = (width / 2) + pan.x
            const cy = (height / 2) + pan.y

            // Scale: pixels per normalized unit
            const scale = (Math.min(width, height) / (2 * BASE_VIEW_RADIUS_NORM)) * zoom

            // Clear
            ctx.fillStyle = theme === 'dark' ? '#0f172a' : '#f8fafc'
            ctx.fillRect(0, 0, width, height)

            // Helper to transform normalized coords (relative to Moon) to canvas coords
            const toCanvas = (x: number, y: number) => {
                return {
                    x: cx + x * scale,
                    y: cy - y * scale // Flip Y for canvas
                }
            }

            // Draw Grid
            ctx.strokeStyle = theme === 'dark' ? '#1e293b' : '#e2e8f0'
            ctx.lineWidth = 1
            const gridSizeKM = 10000
            const gridSizeNorm = gridSizeKM / L_STAR
            // Visible range estimation for grid optimization
            const visibleRadius = (Math.max(width, height) / scale)
            const steps = Math.ceil(visibleRadius / gridSizeNorm)

            ctx.beginPath()
            // Vertical lines
            for (let i = -steps; i <= steps; i++) {
                const x = i * gridSizeNorm
                // Simplified drawing: just draw long lines
                const p = toCanvas(x, 0)
                ctx.moveTo(p.x, -10000) // Draw off-screen limits
                ctx.lineTo(p.x, 10000)
            }
            // Horizontal lines
            for (let i = -steps; i <= steps; i++) {
                const y = i * gridSizeNorm
                const p = toCanvas(0, y)
                ctx.moveTo(-10000, p.y)
                ctx.lineTo(10000, p.y)
            }
            ctx.stroke()

            // Draw Moon (Centered at 0,0 relative coords)
            const moonPosCanvas = toCanvas(0, 0)
            const moonRadiusPx = R_MOON_NORM * scale

            ctx.beginPath()
            ctx.arc(moonPosCanvas.x, moonPosCanvas.y, Math.max(2, moonRadiusPx), 0, Math.PI * 2)
            ctx.fillStyle = '#64748b'
            ctx.fill()

            // Moon Label
            ctx.fillStyle = theme === 'dark' ? '#94a3b8' : '#475569'
            ctx.font = '10px monospace'
            ctx.fillText('MOON', moonPosCanvas.x + moonRadiusPx + 4, moonPosCanvas.y + 3)

            // Draw SOI / Capture Radius (~20,000 km reference)
            const soiRadiusNorm = 20000 / L_STAR
            const soiRadiusPx = soiRadiusNorm * scale
            ctx.beginPath()
            ctx.arc(moonPosCanvas.x, moonPosCanvas.y, soiRadiusPx, 0, Math.PI * 2)
            ctx.strokeStyle = theme === 'dark' ? '#334155' : '#cbd5e1'
            ctx.setLineDash([5, 5])
            ctx.stroke()
            ctx.setLineDash([])

            // Draw Trajectory
            const bestTraj = displayResult?.bestTrajectory
            if (bestTraj && bestTraj.length > 0) {
                ctx.beginPath()
                // Moon position in global frame is [1-MU, 0]
                // We need to shift trajectory points by -(1-MU) to make them relative to Moon
                const shiftX = -(1 - MU)

                // Optimization: Skip rendering if zoom is very low and path is tiny? No, keep it.
                // Optimization: Filter points?

                let first = true
                bestTraj.forEach((p) => {
                    const relX = p[0] + shiftX
                    const relY = p[1]

                    const pt = toCanvas(relX, relY)
                    // Simple clipping check
                    if (pt.x > -100 && pt.x < width + 100 && pt.y > -100 && pt.y < height + 100) {
                        if (first) {
                            ctx.moveTo(pt.x, pt.y)
                            first = false
                        } else {
                            ctx.lineTo(pt.x, pt.y)
                        }
                    } else {
                        // Even if point is out, we might need to draw line TO it if previous was in.
                        // For simplicity in this logic, we skip 'moveTo' on re-entry which creates gaps.
                        // Better: just draw all. Canvas handles clipping efficiently.
                        if (first) ctx.moveTo(pt.x, pt.y); else ctx.lineTo(pt.x, pt.y);
                        first = false;
                    }
                })

                ctx.strokeStyle = getMethodColor(displayMethod)
                ctx.lineWidth = 2
                ctx.stroke()
            }
        }

        render() // Initial render

        return () => window.removeEventListener('resize', resizeHelper)
    }, [results, currentMethod, selectedResult, theme, zoom, pan])

    return (
        <div className="w-full h-full flex flex-col relative text-theme-primary">
            <div className="p-2 border-b border-theme bg-theme-secondary flex justify-between items-center select-none">
                <h3 className="text-sm font-semibold text-theme-header">Lunar Flyby Analysis (2D)</h3>
                <div className="flex items-center gap-2">
                    <span className="text-xs text-theme-muted">{(zoom * 100).toFixed(0)}%</span>
                    <button onClick={() => setZoom(z => Math.max(0.1, z - 0.2))} className="p-1 hover:bg-theme-tertiary rounded">
                        <ZoomOut className="w-3 h-3" />
                    </button>
                    <button onClick={() => setZoom(z => Math.min(10, z + 0.2))} className="p-1 hover:bg-theme-tertiary rounded">
                        <ZoomIn className="w-3 h-3" />
                    </button>
                    <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }) }} className="p-1 hover:bg-theme-tertiary rounded" title="Reset View">
                        <Move className="w-3 h-3" />
                    </button>
                </div>
            </div>

            <div className="flex-1 relative overflow-hidden bg-theme-primary cursor-move">
                <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full block"
                    onWheel={handleWheel}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                />

                {/* Overlay Hint */}
                <div className="absolute bottom-2 right-2 text-[10px] text-theme-muted opacity-50 pointer-events-none select-none">
                    Scroll to Zoom • Drag to Pan
                </div>
            </div>
        </div>
    )
}

function getMethodColor(method: string | undefined): string {
    switch (method) {
        case 'thrml': return '#8b5cf6'
        case 'quantum': return '#06b6d4'
        case 'random': return '#f59e0b'
        default: return '#8b5cf6'
    }
}
