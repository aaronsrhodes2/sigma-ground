# FACE_PORTAL_HANDOFF: the OPTIONAL "link my game account" for the head picture

**Priority: optional, lower than everything else.** The main path needs no portal: every player's head defaults to their public
Steam avatar (fetched by the bridge). This link only exists so someone (Aaron) can override that with the picture of their Google
account. Everything on the game and bridge side is built and tested against the contract below; until the portal side exists the
bridge answers "unavailable" to a link code and `skippy face <https link>` is the manual route.

## Ready-to-paste prompt for the SkippyTel (portal) session

> Add an optional "Link my game account" feature to the portal (signed in with Google). Do not touch the game repo.
>
> 1. **Issue a code.** A signed-in user presses "Link my game account" (POST `/api/game-link/code`, session-authenticated, CSRF
>    protected). The server creates a random single-use code of 8 characters from an unambiguous alphabet, shown as `ABCD-EFGH`
>    (stored without the dash, compared case-insensitively), bound to that user's Google account, with a **10 minute** expiry. The
>    page shows the code and the exact words to type in the game: `skippy link ABCD-EFGH`. At most **one live code per user** (a new
>    one replaces the old) and at most **5 codes per user per hour**.
> 2. **Redeem endpoint (called by the game's bridge, not by browsers).** `POST /api/game-link/redeem`, JSON `{"code": "ABCDEFGH"}`
>    (the dash and case are ignored), over Tailscale only (bind or firewall it to the tailnet; never public). Authenticated by a
>    shared secret in the header `X-Game-Link-Secret` (constant-time compare; the secret comes from `D:\Aaron\development\.env`
>    as `GAME_LINK_SECRET`, the same value the bridge reads as `SKIPPY_PORTAL_SECRET`). Response `200`:
>    `{"ok": true, "picture_url": "https://lh3.googleusercontent.com/a/...=s256", "display_name": "Aaron"}` where `picture_url` is
>    the `picture` claim of the user's Google ID token with its size parameter set to `=s256` (replace a trailing `=s96-c`), and
>    `display_name` is optional. A code is **single use**: marking it used happens in the same transaction that returns the
>    answer, so a second redeem fails. Failures: `404` unknown, `410` expired or already used, `429` too many tries, `401` bad
>    secret. The body of a failure is `{"ok": false}` and says nothing about why (no oracle for guessing codes).
> 3. **Abuse limits.** Redeem is limited to **10 attempts per minute overall** and **3 failed attempts per minute per caller**; codes
>    have 40 bits of randomness or more. No game identity (no SteamID) is ever sent to or stored by the portal: the bridge is the only
>    thing that pairs a code with a game account, and the pairing lives on the game server.
> 4. **Consent and revocation.** The page says plainly: "your Google profile picture will be shown as your head to everyone on the
>    game server; say `skippy face remove` (or `hide`/`off`) in the game any time". Nothing else from the Google account (email, name
>    beyond the optional display name) is exposed. A "Forget my linked game picture" button is nice to have; the game side stores only
>    a 128x128 JPEG on the PC and deletes it when the player removes it.
> 5. **Tests.** Code issue (format, expiry, replacement, hourly cap), redeem (success once, second use 410, expiry 410, wrong secret
>    401, rate limits 429), and that no response includes the email or the Google subject id.

## The contract the bridge expects (two lines)

`POST {SKIPPY_PORTAL_URL}/api/game-link/redeem` with header `X-Game-Link-Secret: {SKIPPY_PORTAL_SECRET}` and JSON `{"code": "..."}`
returns `200 {"ok": true, "picture_url": "https://<googleusercontent host>/...", "display_name": "..."}` or a `4xx` with `{"ok": false}`.
The bridge then fetches `picture_url` itself (https, allow-listed host `*.googleusercontent.com`, public addresses only, 2 MB cap),
shrinks it to 128x128 and hands the bytes to the game; the code is never kept.

## Game side, for reference

In the game the player types `skippy link ABCD-EFGH` (chat; the message is swallowed, so nobody else sees the code) or the console
command `skippy_link ABCD-EFGH`. The game server calls the bridge `POST /face {"action": "redeem", "code": ...}`
(`stage/gmod/addon/lua/mentat/faces_server.lua`, `Faces.link`), which calls the portal as above. A redeemed picture is the player's
own explicit choice (source `set`): the automatic Steam default never replaces it. The stub is `portal_redeem()` in
`stage/gmod/holodeck/skippy_faces.py`; environment: `SKIPPY_PORTAL_URL`, `SKIPPY_PORTAL_SECRET`.
