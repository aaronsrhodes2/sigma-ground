# Quiet mode helper: minimize the Garry's Mod window as soon as it exists, WITHOUT activating it, and keep it
# minimized for the first minute (the game's launcher ignores `start /min`, and the engine restores/raises its
# window while loading). SW_SHOWMINNOACTIVE = 7.
Add-Type -Namespace Win -Name H -MemberDefinition '[DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr h, int cmd); [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);'
$deadline = (Get-Date).AddSeconds(75)
while ((Get-Date) -lt $deadline) {
  $p = Get-Process gmod, gmod_win64 -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowHandle -ne 0 }
  foreach ($q in $p) { if (-not [Win.H]::IsIconic($q.MainWindowHandle)) { [void][Win.H]::ShowWindowAsync($q.MainWindowHandle, 7) } }
  Start-Sleep -Milliseconds 300
}
