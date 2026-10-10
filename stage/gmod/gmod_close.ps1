# Ask the Garry's Mod window to close the way its title-bar X does (WM_SYSCOMMAND / SC_CLOSE).
# Verified 2026-10-10: a plain WM_CLOSE (Process.CloseMainWindow) is IGNORED by the engine for
# 30 s; SC_CLOSE ends the whole process tree cleanly in under 10 s. Exit code 0 = message posted,
# 1 = no GMod window found.
Add-Type -Namespace Win -Name U -MemberDefinition '[DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);'
$p = Get-Process gmod, gmod_win64 -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 }
if (-not $p) { Write-Output 'no GMod window'; exit 1 }
foreach ($q in $p) {
  $ok = [Win.U]::PostMessage($q.MainWindowHandle, 0x0112, [IntPtr]0xF060, [IntPtr]::Zero)
  Write-Output ("SC_CLOSE -> pid " + $q.Id + " (" + $q.ProcessName + ") posted=" + $ok)
}
exit 0
