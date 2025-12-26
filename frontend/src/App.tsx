import { useState, useEffect } from 'react'
import TitleBar from './components/layout/TitleBar'
import Sidebar from './components/layout/Sidebar'
import MainContent from './components/layout/MainContent'
import { useSimulationStore } from './stores/simulationStore'
import { initializeTheme } from './stores/themeStore'
import { ErrorBoundary } from './components/common/ErrorBoundary'

function App() {
    const { setBackendStatus, setBackendPort, errorMessage, setError } = useSimulationStore()
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        // Initialize theme
        initializeTheme()

        // Check backend status on mount
        const checkBackend = async () => {
            try {
                const port = await window.electronAPI?.getBackendPort() || 8080
                setBackendPort(port)

                const status = await window.electronAPI?.getBackendStatus()
                setBackendStatus(status?.status === 'online' ? 'online' : 'offline')
            } catch (error) {
                setBackendStatus('offline')
                setError('Backend unreachable. Please start the bundled backend or run `python server.py` in THRML-Sandbox/backend.')
            } finally {
                setIsLoading(false)
            }
        }

        // Initial check after delay (backend startup time)
        const timer = setTimeout(checkBackend, 1000)

        // Periodic check
        const interval = setInterval(checkBackend, 5000)

        return () => {
            clearTimeout(timer)
            clearInterval(interval)
        }
    }, [setBackendStatus, setBackendPort])

    if (isLoading) {
        return (
            <div className="h-screen bg-theme-primary flex flex-col">
                <TitleBar />
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center fade-in">
                        <div className="w-20 h-20 mx-auto mb-6 relative">
                            <div className="absolute inset-0 rounded-full border-4 border-theme-tertiary"></div>
                            <div className="absolute inset-0 rounded-full border-4 border-t-[var(--accent-blue)] animate-spin"></div>
                            <div className="absolute inset-3 rounded-full bg-gradient-to-br from-[var(--accent-blue)] to-[var(--accent-purple)] opacity-20"></div>
                        </div>
                        <h2 className="text-xl font-semibold text-theme-primary mb-2">Initializing ASL-Sandbox</h2>
                        <p className="text-theme-secondary">Starting physics backend...</p>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <ErrorBoundary>
            <div className="h-screen bg-theme-primary flex flex-col overflow-hidden relative">
                <TitleBar />
                <div className="flex-1 flex overflow-hidden">
                    <Sidebar />
                    <MainContent />
                </div>
                {errorMessage && (
                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20">
                        <div className="glass px-4 py-3 rounded-lg flex items-center gap-3 text-sm shadow-lg">
                            <div className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: 'var(--accent-red)' }}></div>
                            <span className="text-theme-primary">{errorMessage}</span>
                            <button
                                onClick={() => setError(null)}
                                className="ml-2 text-theme-secondary hover:text-theme-primary text-xs"
                            >
                                Dismiss
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </ErrorBoundary>
    )
}

export default App
