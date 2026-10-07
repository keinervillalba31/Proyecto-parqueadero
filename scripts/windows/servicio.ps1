<#
Instala el servicio de monitoreo para que arranque solo (sin abrir una consola
ni ejecutar scripts), usando el Programador de tareas de Windows.

  .\scripts\windows\servicio.ps1 instalar     # se inicia al iniciar sesión y se reinicia si falla
  .\scripts\windows\servicio.ps1 iniciar
  .\scripts\windows\servicio.ps1 detener
  .\scripts\windows\servicio.ps1 estado
  .\scripts\windows\servicio.ps1 desinstalar

Para que arranque al encender el equipo, aunque nadie inicie sesión, ejecútalo
como administrador con:  instalar -AlEncender
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("instalar", "iniciar", "detener", "estado", "desinstalar")]
    [string]$Accion,
    [switch]$AlEncender
)

$ErrorActionPreference = "Stop"
$Nombre = "ParqueaderoServicio"
$Raiz = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $Raiz ".venv\Scripts\pythonw.exe"

switch ($Accion) {
    "instalar" {
        if (-not (Test-Path $Python)) {
            throw "No existe $Python. Crea el entorno: py -3.12 -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt"
        }
        $comando = New-ScheduledTaskAction -Execute $Python -Argument "run_service.py" -WorkingDirectory $Raiz
        $disparador = if ($AlEncender) { New-ScheduledTaskTrigger -AtStartup } else { New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME }
        $ajustes = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
            -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
        Register-ScheduledTask -TaskName $Nombre -Action $comando -Trigger $disparador -Settings $ajustes -Force | Out-Null
        Write-Output "Servicio instalado. Se iniciará solo; para arrancarlo ya: .\scripts\windows\servicio.ps1 iniciar"
    }
    "iniciar"     { Start-ScheduledTask -TaskName $Nombre; Write-Output "Iniciado" }
    "detener"     { Stop-ScheduledTask -TaskName $Nombre; Write-Output "Detenido" }
    "estado"      { Get-ScheduledTask -TaskName $Nombre | Select-Object TaskName, State }
    "desinstalar" { Unregister-ScheduledTask -TaskName $Nombre -Confirm:$false; Write-Output "Servicio eliminado" }
}
