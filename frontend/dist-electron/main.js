"use strict";
const electron = require("electron");
const child_process = require("child_process");
const path = require("path");
const fs = require("fs");
const url = require("url");
var _documentCurrentScript = typeof document !== "undefined" ? document.currentScript : null;
const __dirname$1 = path.dirname(url.fileURLToPath(typeof document === "undefined" ? require("url").pathToFileURL(__filename).href : _documentCurrentScript && _documentCurrentScript.tagName.toUpperCase() === "SCRIPT" && _documentCurrentScript.src || new URL("main.js", document.baseURI).href));
let pythonProcess = null;
const BACKEND_PORT = 8080;
let mainWindow = null;
const isDev = !electron.app.isPackaged;
function log(message) {
  console.log(`[ASL-Sandbox] ${message}`);
  mainWindow == null ? void 0 : mainWindow.webContents.executeJavaScript(
    `console.log('[Main] ${message.replace(/'/g, "\\'")}')`
  ).catch(() => {
  });
}
function getBackendPath() {
  log(`isDev: ${isDev}`);
  log(`__dirname: ${__dirname$1}`);
  const resourcesPath = isDev ? path.join(__dirname$1, "../resources") : process.resourcesPath;
  log(`resourcesPath: ${resourcesPath}`);
  const bundledBackend = path.join(resourcesPath, "backend", "asl-sandbox-backend.exe");
  log(`Looking for bundled backend at: ${bundledBackend}`);
  if (fs.existsSync(bundledBackend)) {
    log("✅ Found bundled Python backend");
    return {
      executable: bundledBackend,
      args: ["--host", "127.0.0.1", "--port", BACKEND_PORT.toString()],
      cwd: path.dirname(bundledBackend),
      found: true
    };
  }
  log("❌ Bundled backend not found");
  const devProjectRoot = isDev ? path.join(__dirname$1, "../..") : path.join(resourcesPath, "../..");
  const backendDir = path.join(devProjectRoot, "THRML-Sandbox", "backend");
  log(`Looking for dev backend at: ${backendDir}`);
  if (fs.existsSync(path.join(backendDir, "server.py"))) {
    log("✅ Found development Python backend");
    return {
      executable: "python",
      args: [
        "-m",
        "uvicorn",
        "server:app",
        "--host",
        "127.0.0.1",
        "--port",
        BACKEND_PORT.toString()
      ],
      cwd: backendDir,
      found: true
    };
  }
  log("❌ No backend found!");
  return {
    executable: "",
    args: [],
    cwd: "",
    found: false
  };
}
function createWindow() {
  mainWindow = new electron.BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    frame: false,
    backgroundColor: "#0a0a0f",
    webPreferences: {
      preload: path.join(__dirname$1, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true
    },
    show: false
  });
  mainWindow.once("ready-to-show", () => {
    mainWindow == null ? void 0 : mainWindow.show();
  });
  if (isDev) {
    mainWindow.loadURL("http://localhost:5173");
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    mainWindow.loadFile(path.join(__dirname$1, "../dist/index.html"));
  }
  mainWindow.webContents.setWindowOpenHandler(({ url: url2 }) => {
    electron.shell.openExternal(url2);
    return { action: "deny" };
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}
function startPythonBackend() {
  var _a, _b;
  const backendConfig = getBackendPath();
  if (!backendConfig.found) {
    log("Cannot start backend - no backend found");
    return;
  }
  log("Starting Python backend...");
  log(`Executable: ${backendConfig.executable}`);
  log(`Args: ${backendConfig.args.join(" ")}`);
  log(`CWD: ${backendConfig.cwd}`);
  try {
    const useShell = !backendConfig.executable.endsWith(".exe");
    pythonProcess = child_process.spawn(backendConfig.executable, backendConfig.args, {
      cwd: backendConfig.cwd,
      shell: useShell,
      detached: false,
      windowsHide: true,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1"
      },
      stdio: ["pipe", "pipe", "pipe"]
    });
    (_a = pythonProcess.stdout) == null ? void 0 : _a.on("data", (data) => {
      const output = data.toString().trim();
      if (output) {
        log(`Backend stdout: ${output}`);
      }
    });
    (_b = pythonProcess.stderr) == null ? void 0 : _b.on("data", (data) => {
      const output = data.toString().trim();
      if (output) {
        log(`Backend stderr: ${output}`);
      }
    });
    pythonProcess.on("error", (err) => {
      log(`Backend spawn error: ${err.message}`);
    });
    pythonProcess.on("close", (code) => {
      log(`Backend exited with code ${code}`);
      pythonProcess = null;
    });
    log(`Backend process started with PID: ${pythonProcess.pid}`);
  } catch (error) {
    log(`Failed to spawn backend: ${error.message}`);
  }
}
function stopPythonBackend() {
  if (pythonProcess) {
    log("Stopping Python backend...");
    if (process.platform === "win32" && pythonProcess.pid) {
      try {
        child_process.spawn("taskkill", ["/pid", pythonProcess.pid.toString(), "/f", "/t"], {
          shell: true
        });
      } catch (e) {
        pythonProcess.kill();
      }
    } else {
      pythonProcess.kill("SIGTERM");
    }
    pythonProcess = null;
  }
}
async function checkBackendHealth() {
  try {
    const response = await fetch(`http://127.0.0.1:${BACKEND_PORT}/`, {
      signal: AbortSignal.timeout(2e3)
    });
    return response.ok;
  } catch {
    return false;
  }
}
electron.ipcMain.handle("window:minimize", () => {
  mainWindow == null ? void 0 : mainWindow.minimize();
});
electron.ipcMain.handle("window:maximize", () => {
  if (mainWindow == null ? void 0 : mainWindow.isMaximized()) {
    mainWindow.unmaximize();
  } else {
    mainWindow == null ? void 0 : mainWindow.maximize();
  }
});
electron.ipcMain.handle("window:close", () => {
  mainWindow == null ? void 0 : mainWindow.close();
});
electron.ipcMain.handle("backend:status", async () => {
  const isOnline = await checkBackendHealth();
  if (isOnline) {
    try {
      const response = await fetch(`http://127.0.0.1:${BACKEND_PORT}/`);
      const data = await response.json();
      return { status: "online", data };
    } catch {
      return { status: "online", data: {} };
    }
  }
  return { status: "offline", error: "Backend not responding" };
});
electron.ipcMain.handle("backend:port", () => {
  return BACKEND_PORT;
});
electron.ipcMain.handle("backend:restart", async () => {
  stopPythonBackend();
  await new Promise((resolve) => setTimeout(resolve, 1e3));
  startPythonBackend();
  return { status: "restarting" };
});
electron.ipcMain.handle("backend:debug", () => {
  const backendConfig = getBackendPath();
  return {
    isDev,
    resourcesPath: isDev ? path.join(__dirname$1, "../resources") : process.resourcesPath,
    ...backendConfig,
    processRunning: pythonProcess !== null,
    processPid: pythonProcess == null ? void 0 : pythonProcess.pid
  };
});
electron.app.whenReady().then(async () => {
  log("App ready, starting backend...");
  const alreadyOnline = await checkBackendHealth();
  if (alreadyOnline) {
    log("Backend already running, skipping spawn");
  } else {
    startPythonBackend();
  }
  let attempts = 0;
  const maxAttempts = 30;
  while (attempts < maxAttempts) {
    const isReady = await checkBackendHealth();
    if (isReady) {
      log("Backend is ready!");
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
    attempts++;
    if (attempts % 5 === 0) {
      log(`Still waiting for backend... (${attempts}/${maxAttempts})`);
    }
  }
  if (attempts >= maxAttempts) {
    log("Backend did not start in time");
  }
  createWindow();
  electron.app.on("activate", () => {
    if (electron.BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});
electron.app.on("window-all-closed", () => {
  stopPythonBackend();
  if (process.platform !== "darwin") {
    electron.app.quit();
  }
});
electron.app.on("before-quit", () => {
  stopPythonBackend();
});
const gotTheLock = electron.app.requestSingleInstanceLock();
if (!gotTheLock) {
  electron.app.quit();
} else {
  electron.app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}
