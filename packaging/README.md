# Packaging notes for Herbivora (maintainers)

## End-user installers (preferred)

| Artifact | Builder | Notes |
|----------|---------|-------|
| `Herbivora-Setup-vVERSION.exe` | [`build_windows_setup.bat`](build_windows_setup.bat) + Inno Setup 6 | Unpacks source and runs `Install_Herbivora.bat`. License page uses [`installer_license.txt`](installer_license.txt) (built from `LICENSE` + `THIRD_PARTY_NOTICES.md`). |
| `Herbivora-vVERSION.dmg` | [`build_macos_dmg.sh`](build_macos_dmg.sh) via **GitHub Actions** (or any Mac) | Drag-to-Applications `Herbivora.app` with the Herbivora leaf icon; first launch performs setup inside the app |

Core logic: [`bootstrap_install.py`](bootstrap_install.py) (GPU detect, private Python on Windows, venv, Torch, deps, models, shortcuts).

Windows private Python helper: [`ensure_windows_python.ps1`](ensure_windows_python.ps1).

Attach **only** these small bootstraps to GitHub Releases. Torch and models download at install time.

### macOS DMG without a local Mac

You **cannot** build a native `.dmg` on Windows (`hdiutil` is macOS-only). Use CI:

1. Publish a GitHub Release for the version tag (with `Herbivora-Setup-v*.exe` if you have it).
2. Workflow [`.github/workflows/macos-dmg.yml`](../.github/workflows/macos-dmg.yml) runs on `macos-latest`, builds `Herbivora-vVERSION.dmg`, and uploads it to that Release.
3. The same workflow uploads **`SHA256SUMS`** (hashes of the DMG and any Setup.exe already on the Release).

Manual re-run: **Actions → macOS DMG → Run workflow** and enter the tag (e.g. `v1.3.5`).

Optional local build (only if you have a Mac):

```bash
chmod +x packaging/build_macos_dmg.sh
./packaging/build_macos_dmg.sh
```

The disk image opens with `Herbivora.app` beside an Applications shortcut. The
user drags the app across, then launches it from Applications; no `.command`
file or Terminal step is part of the normal macOS flow.

### Signing and notarization

Changing the DMG layout does not itself bypass Gatekeeper. A public build must
be signed with an Apple **Developer ID Application** certificate and notarized
to avoid the “Apple could not verify” warning on a normal double-click.

Add these GitHub Actions secrets to enable the existing macOS workflow:

| Secret | Value |
|--------|-------|
| `MACOS_CERTIFICATE` | Developer ID Application `.p12`, base64-encoded |
| `MACOS_CERTIFICATE_PASSWORD` | Password used when exporting the `.p12` |
| `MACOS_SIGNING_IDENTITY` | Full identity, e.g. `Developer ID Application: Name (TEAMID)` |
| `APPLE_ID` | Apple Developer account email |
| `APPLE_TEAM_ID` | 10-character Apple Developer team ID |
| `APPLE_APP_PASSWORD` | App-specific password for notarization |

With all secrets present, `.github/workflows/macos-dmg.yml` imports the
certificate, signs the app and DMG, submits the DMG to Apple, staples the
notarization ticket, and verifies Gatekeeper acceptance before upload.

Without them the workflow still produces a functional ad-hoc-signed DMG, and
that build **is blocked on every Mac except the one that built it**. Check any
candidate build before sending it to a tester:

```bash
syspolicy_check distribution dist/dmg_stage/Herbivora.app
```

`Notary Ticket Missing / Severity: Fatal` means recipients will be stopped. Two
different alerts follow, both from the same cause:

| How the recipient launched it | Alert | Recoverable? |
|---|---|---|
| Double-click inside the mounted DMG | *The application "Herbivora.app" can't be opened.* (OK only) | **No.** Dead end; they must copy it to `/Applications` first |
| Double-click in `/Applications` | *Apple could not verify "Herbivora" is free of malware* | Yes, via **System Settings → Privacy & Security → Open Anyway** |

The DMG ships ` READ ME FIRST.txt` (generated from
[`macos_app/dmg_readme.txt`](macos_app/dmg_readme.txt), leading space so Finder
sorts it first) covering both. Right-click → **Open** is no longer a reliable
bypass; macOS 15 removed it for unnotarized apps, so the instructions lead with
the Privacy & Security route and offer
`xattr -dr com.apple.quarantine /Applications/Herbivora.app` as the one-command
alternative.

### Verify integrity

```bash
# macOS / Linux
shasum -a 256 -c SHA256SUMS
```

```powershell
# Windows (PowerShell) — compare to the line in SHA256SUMS
Get-FileHash .\Herbivora-Setup-v1.3.5.exe -Algorithm SHA256
```

## Optional PyInstaller (not for normal Releases)

Large onedir bundles; easy to exceed GitHub’s 2 GB asset limit.

```bat
pip install -r requirements-dev.txt
REM Ensure the active venv already has the desired torch (CPU or CUDA), then:
packaging\build_windows.bat
```

```bash
chmod +x packaging/build_macos.sh
./packaging/build_macos.sh
```

## Models

Weights are never bundled. Use the bootstrap installer, **Check installation**, or `download_models.py`.
