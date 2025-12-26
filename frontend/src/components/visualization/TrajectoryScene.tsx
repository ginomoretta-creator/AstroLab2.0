import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import { useSimulationStore } from '../../stores/simulationStore'
import { useThemeStore } from '../../stores/themeStore'

// Constants (normalized units - Earth at origin, Moon at ~1.0)
const MU = 0.01215058560962404

// Apply different visualization scales for each body
const EARTH_SCALE = 1.2   // Earth slightly larger than true scale
const MOON_SCALE = 3.0    // Moon needs more scaling to be visible
const EARTH_RADIUS = 0.01659 * EARTH_SCALE
const MOON_RADIUS = 0.00452 * MOON_SCALE
const MOON_POSITION: [number, number, number] = [1 - MU, 0, 0]

export default function TrajectoryScene() {
    const { results, currentMethod, trajectoryHistory, selectedResult } = useSimulationStore()
    const { theme } = useThemeStore()

    // Use selected result (history) or current result
    const displayResult = selectedResult || results[currentMethod]
    // Use method from selected result if available (default to currentMethod if live)
    const displayMethod = selectedResult?.method || currentMethod

    return (
        <group>
            {/* Earth */}
            <Earth theme={theme} />

            {/* Moon */}
            <Moon theme={theme} />

            {/* Orbit reference circle */}
            <OrbitReference theme={theme} />

            {/* Historical trajectories (faded background) */}
            {trajectoryHistory.slice(-20).map((traj, index) => (
                <TrajectoryPath
                    key={index}
                    points={traj.points}
                    color={getMethodColor(traj.method)}
                    opacity={0.4}
                    simple={true}
                />
            ))}

            {/* Best trajectory highlight (Selected or Current) */}
            {displayResult?.bestTrajectory && displayResult.bestTrajectory.length > 0 && (
                <TrajectoryPath
                    points={displayResult.bestTrajectory.map(p => [p[0], p[1]] as [number, number])}
                    color={getMethodColor(displayMethod)}
                    opacity={1}
                    lineWidth={3}
                    glow={false}
                />
            )}

            {/* Grid for reference */}
            <gridHelper
                args={[4, 20, theme === 'dark' ? '#333' : '#ccc', theme === 'dark' ? '#222' : '#eee']}
                rotation={[Math.PI / 2, 0, 0]}
            />
        </group>
    )
}

interface PlanetProps {
    theme: 'dark' | 'light'
}

function Earth({ theme }: PlanetProps) {
    const meshRef = useRef<THREE.Mesh>(null)

    useFrame((_, delta) => {
        if (meshRef.current) {
            meshRef.current.rotation.y += delta * 0.3
        }
    })

    const earthColor = theme === 'dark' ? '#3b82f6' : '#2563eb'
    const emissiveColor = theme === 'dark' ? '#1e40af' : '#1d4ed8'
    const glowColor = theme === 'dark' ? '#60a5fa' : '#93c5fd'

    return (
        <group position={[-MU, 0, 0]}>
            <mesh ref={meshRef}>
                <sphereGeometry args={[EARTH_RADIUS, 32, 32]} />
                <meshStandardMaterial
                    color={earthColor}
                    emissive={emissiveColor}
                    emissiveIntensity={theme === 'dark' ? 0.3 : 0.1}
                />
            </mesh>
            <mesh scale={1.15}>
                <sphereGeometry args={[EARTH_RADIUS, 32, 32]} />
                <meshBasicMaterial
                    color={glowColor}
                    transparent
                    opacity={theme === 'dark' ? 0.15 : 0.08}
                    side={THREE.BackSide}
                />
            </mesh>
        </group>
    )
}

function Moon({ theme }: PlanetProps) {
    const meshRef = useRef<THREE.Mesh>(null)

    useFrame((_, delta) => {
        if (meshRef.current) {
            meshRef.current.rotation.y += delta * 0.1
        }
    })

    const moonColor = theme === 'dark' ? '#94a3b8' : '#64748b'

    return (
        <group position={MOON_POSITION}>
            <mesh ref={meshRef}>
                <sphereGeometry args={[MOON_RADIUS, 32, 32]} />
                <meshStandardMaterial
                    color={moonColor}
                    emissive={moonColor}
                    emissiveIntensity={theme === 'dark' ? 0.1 : 0.05}
                />
            </mesh>
        </group>
    )
}

function OrbitReference({ theme }: PlanetProps) {
    const points = useMemo(() => {
        const pts: [number, number, number][] = []
        for (let i = 0; i <= 64; i++) {
            const angle = (i / 64) * Math.PI * 2
            pts.push([-MU + Math.cos(angle) * 0.5, Math.sin(angle) * 0.5, 0])
        }
        return pts
    }, [])

    const lineColor = theme === 'dark' ? '#444' : '#ccc'

    return (
        <Line points={points} color={lineColor} lineWidth={1} transparent opacity={0.3} />
    )
}

interface TrajectoryPathProps {
    points: [number, number][]
    color: string
    opacity?: number
    lineWidth?: number
    glow?: boolean
    simple?: boolean
}

function TrajectoryPath({ points, color, opacity = 1, lineWidth = 2, simple = false }: TrajectoryPathProps) {
    const geometry = useMemo(() => {
        if (!Array.isArray(points) || points.length < 2) return null
        const validPoints = points.filter(p =>
            Array.isArray(p) && p.length === 2 && Number.isFinite(p[0]) && Number.isFinite(p[1])
        )
        if (validPoints.length < 2) return null

        if (simple) {
            const points3D = validPoints.map(p => new THREE.Vector3(p[0], p[1], 0))
            return new THREE.BufferGeometry().setFromPoints(points3D)
        } else {
            return validPoints.map(p => [p[0], p[1], 0] as [number, number, number])
        }
    }, [points, simple])

    if (!geometry) return null

    if (simple) {
        return (
            <line geometry={geometry as THREE.BufferGeometry}>
                <lineBasicMaterial color={color} opacity={opacity} transparent linewidth={1} />
            </line>
        )
    }

    return (
        <group>
            <Line
                points={geometry as [number, number, number][]}
                color={color}
                lineWidth={lineWidth}
                transparent
                opacity={opacity}
            />
        </group>
    )
}

function getMethodColor(method: string | undefined): string {
    switch (method) {
        case 'thrml': return '#8b5cf6'
        case 'quantum': return '#06b6d4'
        case 'random': return '#f59e0b'
        default: return '#8b5cf6' // Default to purple if undefined
    }
}
