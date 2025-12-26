"use strict";
const electron = require("electron");
electron.contextBridge.exposeInMainWorld("electronAPI", {
  // Window controls
  minimize: () => electron.ipcRenderer.invoke("window:minimize"),
  maximize: () => electron.ipcRenderer.invoke("window:maximize"),
  close: () => electron.ipcRenderer.invoke("window:close"),
  // Backend
  getBackendStatus: () => electron.ipcRenderer.invoke("backend:status"),
  getBackendPort: () => electron.ipcRenderer.invoke("backend:port"),
  restartBackend: () => electron.ipcRenderer.invoke("backend:restart"),
  // Platform info
  platform: process.platform
});
