import { contextBridge, ipcRenderer } from 'electron'

// Expose protected methods to renderer
contextBridge.exposeInMainWorld('electronAPI', {
    // Window controls
    minimize: () => ipcRenderer.invoke('window:minimize'),
    maximize: () => ipcRenderer.invoke('window:maximize'),
    close: () => ipcRenderer.invoke('window:close'),

    // Backend
    getBackendStatus: () => ipcRenderer.invoke('backend:status'),
    getBackendPort: () => ipcRenderer.invoke('backend:port'),
    restartBackend: () => ipcRenderer.invoke('backend:restart'),

    // Platform info
    platform: process.platform,
})

// Type declarations for TypeScript
declare global {
    interface Window {
        electronAPI: {
            minimize: () => Promise<void>
            maximize: () => Promise<void>
            close: () => Promise<void>
            getBackendStatus: () => Promise<{ status: string; data?: any; error?: string }>
            getBackendPort: () => Promise<number>
            restartBackend: () => Promise<{ status: string }>
            platform: string
        }
    }
}
