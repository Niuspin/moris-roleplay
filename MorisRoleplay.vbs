' Silent zero-console desktop launcher for Moris Roleplay
Option Explicit
Dim fso, shell, appDir, appPy, rc
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
appDir = fso.GetParentFolderName(WScript.ScriptFullName)
appPy = appDir & "\app.py"

' Prefer pythonw.exe (no console window), fallback to pyw -3
On Error Resume Next
rc = shell.Run("pythonw """ & appPy & """", 0, False)
If Err.Number <> 0 Then
    Err.Clear
    rc = shell.Run("pyw -3 """ & appPy & """", 0, False)
    If Err.Number <> 0 Then
        MsgBox "未检测到 Python 3.11+ (pythonw/pyw)，请先安装 Python 3.11 或以上版本。", 16, "Moris Roleplay"
    End If
End If
On Error GoTo 0
