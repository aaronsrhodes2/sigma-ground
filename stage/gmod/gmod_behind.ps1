# Test-run helper: keep the Garry's Mod window VISIBLE (so it renders and screenshots work) but BEHIND every other window and NOT active,
# so a test run never takes the keyboard focus or the mouse from whoever is working at the PC. Used by gmod_ctl.sh for non-interactive visible runs
# (screenshots, found-object checks). An interactive play session (GMOD_MOUSE=1, see holodeck/play.py) does not use it.
# SW_SHOWNOACTIVATE = 4; HWND_BOTTOM = 1; SWP_NOSIZE|SWP_NOMOVE|SWP_NOACTIVATE = 0x13.
Add-Type -Namespace Win -Name B -MemberDefinition '[DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr h, int cmd); [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h); [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);'
$bottom = [IntPtr]1
$deadline = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $deadline) {
  $p = Get-Process gmod, gmod_win64 -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 }
  foreach ($q in $p) {
    if ([Win.B]::IsIconic($q.MainWindowHandle)) { [void][Win.B]::ShowWindowAsync($q.MainWindowHandle, 4) }
    [void][Win.B]::SetWindowPos($q.MainWindowHandle, $bottom, 0, 0, 0, 0, 0x13)
  }
  Start-Sleep -Milliseconds 400
}
