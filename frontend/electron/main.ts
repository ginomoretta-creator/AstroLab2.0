import { app, BrowserWindow, ipcMain, shell, dialog } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Backend process
let pythonProcess: ChildProcess | null = null
const BACKEND_PORT = 8080

// Window reference
let mainWindow: BrowserWindow | null = null

// Development mode check
const isDev = !app.isPackaged

// Log file for debugging
function log(message: string) {
    console.log(`[ASL-Sandbox] ${message}`)
    // Also send to renderer if window exists
    mainWindow?.webContents.executeJavaScript(
        `console.log('[Main] ${message.replace(/'/g, "\\'")}')`
    ).catch(() => { })
}

/**
 * Get the path to the Python backend
 */
function getBackendPath(): { executable: string; args: string[]; cwd: string; found: boolean } {
    log(`isDev: ${isDev}`)
    log(`__dirname: ${__dirname}`)

    // In production, resources are in process.resourcesPath
    const resourcesPath = isDev
        ? path.join(__dirname, '../resources')
        : process.resourcesPath

    log(`resourcesPath: ${resourcesPath}`)

    // Check for bundled backend
    const bundledBackend = path.join(resourcesPath, 'backend', 'asl-sandbox-backend.exe')
    log(`Looking for bundled backend at: ${bundledBackend}`)

    if (fs.existsSync(bundledBackend)) {
        log('✅ Found bundled Python backend')
        return {
            executable: bundledBackend,
            args: ['--host', '127.0.0.1', '--port', BACKEND_PORT.toString()],
            cwd: path.dirname(bundledBackend),
            found: true
        }
    }

    log('❌ Bundled backend not found')

    // Try development mode Python script
    // In development, go up from dist-electron to desktop-app to ASL-Sandbox
    const devProjectRoot = isDev
        ? path.join(__dirname, '../..')
        : path.join(resourcesPath, '../..')

    const backendDir = path.join(devProjectRoot, 'THRML-Sandbox', 'backend')
    log(`Looking for dev backend at: ${backendDir}`)

    if (fs.existsSync(path.join(backendDir, 'server.py'))) {
        log('✅ Found development Python backend')
        return {
            executable: 'python',
            args: [
                '-m', 'uvicorn',
                'server:app',
                '--host', '127.0.0.1',
                '--port', BACKEND_PORT.toString(),
            ],
            cwd: backendDir,
            found: true
        }
    }

    log('❌ No backend found!')
    return {
        executable: '',
        args: [],
        cwd: '',
        found: false
    }
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1200,
        minHeight: 700,
        frame: false,
        backgroundColor: '#0a0a0f',
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true,
        },
        show: false,
    })

    mainWindow.once('ready-to-show', () => {
        mainWindow?.show()
    })

    if (isDev) {
        mainWindow.loadURL('http://localhost:5173')
        mainWindow.webContents.openDevTools({ mode: 'detach' })
    } else {
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
    }

    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url)
        return { action: 'deny' }
    })

    mainWindow.on('closed', () => {
        mainWindow = null
    })
}

function startPythonBackend() {
    const backendConfig = getBackendPath()

    if (!backendConfig.found) {
        log('Cannot start backend - no backend found')
        return
    }

    log('Starting Python backend...')
    log(`Executable: ${backendConfig.executable}`)
    log(`Args: ${backendConfig.args.join(' ')}`)
    log(`CWD: ${backendConfig.cwd}`)

    try {
        // For bundled executable, spawn directly without shell
        const useShell = !backendConfig.executable.endsWith('.exe')

        pythonProcess = spawn(backendConfig.executable, backendConfig.args, {
            cwd: backendConfig.cwd,
            shell: useShell,
            detached: false,
            windowsHide: true,
            env: {
                ...process.env,
                PYTHONUNBUFFERED: '1',
            },
            stdio: ['pipe', 'pipe', 'pipe'],
        })

        pythonProcess.stdout?.on('data', (data) => {
            const output = data.toString().trim()
            if (output) {
                log(`Backend stdout: ${output}`)
            }
        })

        pythonProcess.stderr?.on('data', (data) => {
            const output = data.toString().trim()
            if (output) {
                log(`Backend stderr: ${output}`)
            }
        })

        pythonProcess.on('error', (err) => {
            log(`Backend spawn error: ${err.message}`)
        })

        pythonProcess.on('close', (code) => {
            log(`Backend exited with code ${code}`)
            pythonProcess = null
        })

        log(`Backend process started with PID: ${pythonProcess.pid}`)

    } catch (error: any) {
        log(`Failed to spawn backend: ${error.message}`)
    }
}

function stopPythonBackend() {
    if (pythonProcess) {
        log('Stopping Python backend...')

        // On Windows, use taskkill for reliable termination
        if (process.platform === 'win32' && pythonProcess.pid) {
            try {
                spawn('taskkill', ['/pid', pythonProcess.pid.toString(), '/f', '/t'], {
                    shell: true
                })
            } catch (e) {
                pythonProcess.kill()
            }
        } else {
            pythonProcess.kill('SIGTERM')
        }

        pythonProcess = null
    }
}

async function checkBackendHealth(): Promise<boolean> {
    try {
        const response = await fetch(`http://127.0.0.1:${BACKEND_PORT}/`, {
            signal: AbortSignal.timeout(2000)
        })
        return response.ok
    } catch {
        return false
    }
}

// IPC Handlers
ipcMain.handle('window:minimize', () => {
    mainWindow?.minimize()
})

ipcMain.handle('window:maximize', () => {
    if (mainWindow?.isMaximized()) {
        mainWindow.unmaximize()
    } else {
        mainWindow?.maximize()
    }
})

ipcMain.handle('window:close', () => {
    mainWindow?.close()
})

ipcMain.handle('backend:status', async () => {
    const isOnline = await checkBackendHealth()

    if (isOnline) {
        try {
            const response = await fetch(`http://127.0.0.1:${BACKEND_PORT}/`)
            const data = await response.json()
            return { status: 'online', data }
        } catch {
            return { status: 'online', data: {} }
        }
    }

    return { status: 'offline', error: 'Backend not responding' }
})

ipcMain.handle('backend:port', () => {
    return BACKEND_PORT
})

ipcMain.handle('backend:restart', async () => {
    stopPythonBackend()
    await new Promise(resolve => setTimeout(resolve, 1000))
    startPythonBackend()
    return { status: 'restarting' }
})

ipcMain.handle('backend:debug', () => {
    const backendConfig = getBackendPath()
    return {
        isDev,
        resourcesPath: isDev ? path.join(__dirname, '../resources') : process.resourcesPath,
        ...backendConfig,
        processRunning: pythonProcess !== null,
        processPid: pythonProcess?.pid
    }
})

// App lifecycle
app.whenReady().then(async () => {
    log('App ready, starting backend...')

    // Start backend only if it's not already running
    const alreadyOnline = await checkBackendHealth()
    if (alreadyOnline) {
        log('Backend already running, skipping spawn')
    } else {
        startPythonBackend()
    }

    // Wait for backend to be ready
    let attempts = 0
    const maxAttempts = 30 // 15 seconds

    while (attempts < maxAttempts) {
        const isReady = await checkBackendHealth()
        if (isReady) {
            log('Backend is ready!')
            break
        }
        await new Promise(resolve => setTimeout(resolve, 500))
        attempts++
        if (attempts % 5 === 0) {
            log(`Still waiting for backend... (${attempts}/${maxAttempts})`)
        }
    }

    if (attempts >= maxAttempts) {
        log('Backend did not start in time')
    }

    createWindow()

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow()
        }
    })
})

app.on('window-all-closed', () => {
    stopPythonBackend()
    if (process.platform !== 'darwin') {
        app.quit()
    }
})

app.on('before-quit', () => {
    stopPythonBackend()
})

// Handle second instance
const gotTheLock = app.requestSingleInstanceLock()

if (!gotTheLock) {
    app.quit()
} else {
    app.on('second-instance', () => {
        if (mainWindow) {
            if (mainWindow.isMinimized()) mainWindow.restore()
            mainWindow.focus()
        }
    })
}
