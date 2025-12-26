import { Minus, Square, X, Rocket, Wifi, WifiOff, Sun, Moon } from 'lucide-react'
import { useSimulationStore } from '../../stores/simulationStore'
import { useThemeStore } from '../../stores/themeStore'

export default function TitleBar() {
    const { backendStatus } = useSimulationStore()
    const { theme, toggleTheme } = useThemeStore()

    const handleMinimize = () => window.electronAPI?.minimize()
    const handleMaximize = () => window.electronAPI?.maximize()
    const handleClose = () => window.electronAPI?.close()

    return (
        <div className="title-bar h-10 bg-theme-secondary border-b border-theme flex items-center justify-between px-4">
            {/* Logo & Title */}
            <div className="flex items-center gap-3">
                <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-accent-blue to-accent-purple flex items-center justify-center"
                    style={{ background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-purple))' }}>
                    <Rocket className="w-4 h-4 text-white" />
                </div>
                <span className="font-semibold text-sm text-theme-primary">ASL-Sandbox</span>
                <span className="text-xs text-theme-muted">v1.0.0</span>
            </div>

            {/* Center - Backend Status & Theme Toggle */}
            <div className="flex items-center gap-4">
                {/* Backend Status */}
                <div className="flex items-center gap-2">
                    {backendStatus === 'online' ? (
                        <>
                            <Wifi className="w-4 h-4" style={{ color: 'var(--accent-green)' }} />
                            <span className="text-xs" style={{ color: 'var(--accent-green)' }}>Backend Online</span>
                        </>
                    ) : backendStatus === 'offline' ? (
                        <>
                            <WifiOff className="w-4 h-4" style={{ color: 'var(--accent-red)' }} />
                            <span className="text-xs" style={{ color: 'var(--accent-red)' }}>Backend Offline</span>
                        </>
                    ) : (
                        <>
                            <div className="w-4 h-4 border-2 border-theme-muted border-t-transparent rounded-full animate-spin" />
                            <span className="text-xs text-theme-muted">Connecting...</span>
                        </>
                    )}
                </div>

                {/* Theme Toggle */}
                <button
                    onClick={toggleTheme}
                    className="p-1.5 rounded-lg bg-theme-tertiary hover:bg-theme-elevated transition-all"
                    title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
                >
                    {theme === 'dark' ? (
                        <Sun className="w-4 h-4 text-yellow-400" />
                    ) : (
                        <Moon className="w-4 h-4 text-indigo-500" />
                    )}
                </button>
            </div>

            {/* Window Controls */}
            <div className="flex items-center">
                <button
                    onClick={handleMinimize}
                    className="w-10 h-10 flex items-center justify-center hover:bg-theme-tertiary transition-colors"
                >
                    <Minus className="w-4 h-4 text-theme-secondary" />
                </button>
                <button
                    onClick={handleMaximize}
                    className="w-10 h-10 flex items-center justify-center hover:bg-theme-tertiary transition-colors"
                >
                    <Square className="w-3.5 h-3.5 text-theme-secondary" />
                </button>
                <button
                    onClick={handleClose}
                    className="w-10 h-10 flex items-center justify-center hover:bg-red-500 transition-colors group"
                >
                    <X className="w-4 h-4 text-theme-secondary group-hover:text-white" />
                </button>
            </div>
        </div>
    )
}
