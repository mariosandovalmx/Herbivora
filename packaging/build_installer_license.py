#!/usr/bin/env python3
"""Build packaging/installer_license.txt for Windows Setup and installers.

Concatenates LICENSE, citation / commercial terms, and THIRD_PARTY_NOTICES.md
into one plain-text agreement shown on the Inno Setup license page.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "installer_license.txt"

HEADER = """\
================================================================================
HerbivoR — END-USER LICENSE AGREEMENT (Installer copy)
================================================================================

By installing or using HerbivoR you agree to the terms below.
This file is shown by the installer; the same terms ship as LICENSE and
THIRD_PARTY_NOTICES.md in the installation folder.

Copyright (c) 2026 Mario Sandoval

IMPORTANT SUMMARY (not a substitute for the full terms):
  • Noncommercial research, education, and similar noncommercial use: permitted.
  • Commercial use of HerbivoR or HerbivoR-trained weights: NOT permitted without
    prior written permission from the copyright holder.
  • Attribution required when you use HerbivoR or its trained weights in a
    publication, thesis, presentation, or similar work (see CITATION.cff).
  • Third-party components (MobileSAM, BiRefNet, Python packages, etc.) remain
    under their own licenses (see Part C below).
  • Software and weights are provided AS IS, without warranty.

Commercial licensing / permissions:
  https://github.com/mariosandovalmx/HerbivoR

================================================================================
PART A — HerbivoR LICENSE (PolyForm Noncommercial License 1.0.0 + notices)
================================================================================

"""

PART_B = """\

================================================================================
PART B — REQUIRED ATTRIBUTION (CITATION)
================================================================================

If you use HerbivoR (application and/or HerbivoR-trained model weights) in a
publication, thesis, presentation, or similar work, you MUST give appropriate
credit. See CITATION.cff in the HerbivoR distribution for the preferred citation
metadata.

Required Notice: Copyright (c) 2026 Mario Sandoval — HerbivoR (software and
HerbivoR-trained weights)
Required Notice: Noncommercial use only (research / education). Commercial use
requires prior written permission from the copyright holder.
Required Notice: Attribution required — if you use HerbivoR or its trained
weights in a publication or presentation, cite it (see CITATION.cff).

"""

PART_C_HEADER = """\
================================================================================
PART C — THIRD-PARTY NOTICES
================================================================================

"""

FOOTER = """\

================================================================================
END OF INSTALLER LICENSE TEXT
================================================================================

Installing HerbivoR constitutes acceptance of Parts A–C above.
"""


def build_text() -> str:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    third = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    return HEADER + license_text.rstrip() + "\n" + PART_B + PART_C_HEADER + third.rstrip() + "\n" + FOOTER


def main() -> int:
    text = build_text()
    OUT.write_text(text, encoding="utf-8", newline="\r\n")
    print(f"Wrote {OUT} ({len(text)} characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
