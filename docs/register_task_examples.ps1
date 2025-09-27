# Example: Register scheduled task to run run_auto_rule.ps1 every weekday at 08:00
# Requires to be run in an elevated PowerShell session if registering for other users.

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-ExecutionPolicy Bypass -File "C:\Repos\quant_trading_system\scripts\run_auto_rule.ps1" -DryRun'
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Mon,Tue,Wed,Thu,Fri -At 8:00AM
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive -RunLevel LeastPrivilege
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName "AutoRuleDaily" -Action $action -Trigger $trigger -Principal $principal -Settings $settings

# To register using the XML:
# Register-ScheduledTask -Xml (Get-Content 'C:\Repos\quant_trading_system\docs\run_auto_rule_task.xml' | Out-String) -TaskName 'AutoRuleDaily'

# To test run immediately (manual run):
# Start-ScheduledTask -TaskName "AutoRuleDaily"

# To unregister:
# Unregister-ScheduledTask -TaskName "AutoRuleDaily" -Confirm:$false
