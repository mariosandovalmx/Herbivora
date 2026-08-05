HerbivoR __VERSION__ : how to install on macOS
==============================================

Please read step 1 before double-clicking anything. Opening HerbivoR directly
from this disk image window shows a dead-end error ("The application
"HerbivoR.app" can't be opened.") with no way to continue.


STEP 1 : Copy the app out of this window
----------------------------------------
Drag the green HerbivoR leaf icon onto the Applications folder shown next to it.
Then eject this disk image (drag it to the Trash, or press Command-E).


STEP 2 : Open HerbivoR from your Applications folder
----------------------------------------------------
Open Finder, go to Applications, and double-click HerbivoR.

macOS will refuse the first time and show:

    "HerbivoR" Not Opened
    Apple could not verify "HerbivoR" is free of malware that may harm
    your Mac or compromise your privacy.

Click Done. Do NOT click "Move to Trash".

This appears because HerbivoR is distributed without an Apple Developer ID
certificate, which Apple sells to developers for a yearly fee. It is not a
report that anything was found in the app.


STEP 3 : Approve HerbivoR once
------------------------------
1. Open System Settings, then Privacy & Security.
2. Scroll down to the Security section. You will see a line saying
   "HerbivoR" was blocked to protect your Mac.
3. Click Open Anyway and confirm with Touch ID or your password.
4. Double-click HerbivoR in Applications again, then click Open Anyway
   in the confirmation dialog.

HerbivoR now opens normally every time. You only do this once.

On macOS 13 and macOS 14 the shorter path also works: in the Applications
folder, right-click (or Control-click) HerbivoR, choose Open, then click Open
in the dialog.


FIRST LAUNCH TAKES A WHILE
--------------------------
The first launch runs a one-time setup that downloads a private Python
environment, PyTorch, and the model weights. Expect 5 to 20 minutes on a normal
connection, and keep the Mac awake and online. Later launches start the app
directly.


IF YOU PREFER THE TERMINAL
--------------------------
This one command replaces steps 2 and 3 entirely. Run it after step 1:

    xattr -dr com.apple.quarantine /Applications/HerbivoR.app

Then open HerbivoR normally.


STILL STUCK?
------------
The setup log is written to:

    ~/Library/Application Support/HerbivoR/launch.log

Send that file with your report to
https://github.com/mariosandovalmx/HerbivoR/issues
