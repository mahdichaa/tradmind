# Docker Connection Troubleshooting

The error `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified` indicates that the **Docker Engine is not accessible**. This is an issue with the Docker Desktop application on your Windows machine, not the code.

## 1. Verify Docker Desktop Status
- Look at the **Whale icon** in your system tray (near the clock).
- If it is **grey** or **animating**, Docker is not ready.
- It must be **solid white/blue** and say "Docker Desktop is running" when you hover over it.

## 2. Restart Docker Desktop
1. Right-click the Docker icon in the system tray.
2. Select **Quit Docker Desktop**.
3. Open **Docker Desktop** again from the Start Menu.
4. **Wait** (it can take 1-2 minutes) until the bottom left corner of the Docker Desktop window is **Green**.

## 3. Check WSL Integration (Common Issue)
1. Open Command Prompt (cmd) or PowerShell.
2. Run: `wsl --list --verbose`
3. You should see `docker-desktop` and `docker-desktop-data` with STATE `Running`.
   - If they are `Stopped`, Docker Desktop is not running correctly.

## 4. Reset Docker Context
Sometimes the CLI points to the wrong context. Run these commands:
```powershell
docker context use default
```

## 5. Administrator Privileges
Try running your terminal (VS Code or PowerShell) as **Administrator**.

## 6. Reinstall Docker Desktop
If none of the above works, your Docker installation might be corrupted.
1. Uninstall Docker Desktop.
2. Restart your computer.
3. Install the latest version from [docker.com](https://www.docker.com/products/docker-desktop/).
