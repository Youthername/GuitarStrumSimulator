# ChordCraft v0.2 — Interactive Guitar Strum Synth

ChordCraft v0.2 is an interactive guitar strumming synthesizer with a built-in GUI.

You type:
- a chord progression (like `Em C G D`)
- a human strumming pattern (like `D(2) D(1) D(0.5) U(0.5)`)
- tempo (BPM), loops, strum feel, etc.

It will:
1. Generate a readable 6-line guitar TAB.
2. Animate a scrolling TAB preview (silent).
3. Export a realistic guitar-style WAV file using physical string modeling.

This version is designed for stable offline rendering:
it does not rely on your audio driver, and it does not attempt real-time playback.

------------------------------------------------------------
Core ideas
------------------------------------------------------------

- Language: Python 3.8+
- Synthesis: Karplus–Strong plucked string model (physically inspired string decay)
- Output: 16-bit mono WAV @ 48 kHz
- Interface: Tkinter GUI (fully interactive; no terminal flags required)
- Timing: Groove-aware. Each strum hit has its own duration.

Unlike a simple metronome, this engine respects per-hit timing.
For example, `D(2)` lasts longer than `D(0.5)` in the output audio, so you actually hear downbeats lingering and offbeats snapping by.

------------------------------------------------------------
Requirements
------------------------------------------------------------

You only need Python and NumPy.

Dependency / Install:
- Python >= 3.8  (download from python.org if needed)
- NumPy >= 1.21  ->  pip install numpy

tkinter is included with most standard Python builds on Windows/macOS/Linux.
No "simpleaudio", no soundcard hooks — WAV is rendered offline, then you play it in any media player.

------------------------------------------------------------
Project Layout
------------------------------------------------------------

Suggested repo structure:

ChordCraft/
├─ chordcraft_gui_groove.py   # GUI app (main entry point)
├─ README.txt                 # This file

You can add example progressions or screenshots later if you want.

------------------------------------------------------------
How to Run
------------------------------------------------------------

1. Install NumPy:
   pip install numpy

2. Launch the GUI:
   python chordcraft_gui_groove.py

You should now see a window with input fields and buttons.

------------------------------------------------------------
GUI Controls
------------------------------------------------------------

Inputs:

- Chords
  Space-separated chord names, e.g.
  Em C G D
  These chord names must exist in the built-in chord dictionary (Em, C, G, D, Dsus4, etc.).

- Pattern
  Strumming sequence with explicit durations in beats.
  Example (this is the default):
  D(2) D(1) D(0.5) U(0.5)

  Meaning:
  - D = downstroke
  - U = upstroke
  - The number in parentheses = how long that hit lasts in beats
    (e.g. 2 beats, 1 beat, 0.5 beat)

  You can also write patterns like:
  D:2
  D2
  U0.5

- BPM
  Tempo in beats per minute. Affects how long one beat is in seconds.

- SPB
  Steps Per Beat.
  This is only used for drawing the TAB grid and for the silent scroll preview timing.
  SPB does NOT affect final WAV timing in groove mode.

- Loops
  How many times to repeat the chord progression + pattern.

- Window
  How many columns of TAB to show at once in the scrolling preview window.

- Strum ms
  Per-string delay (in milliseconds) within a single strum.
  A higher number (for example 20 ms) makes the chord “sweep” across the strings like a real guitarist instead of all strings firing at once.

Buttons:

- Preview TAB
  Generates a 6-line TAB (standard e|B|G|D|A|E layout) in the text box.

- Scroll Preview (silent)
  Animates a caret "^" moving through the TAB columns in time.
  This uses a uniform timing model (like a drum machine grid).
  It’s meant for visual demo only and does not play sound.

- Stop Scroll
  Stops the caret animation.

- Export WAV (groove)
  This is the main feature.
  It renders actual guitar-like audio using Karplus–Strong synthesis and groove-aware timing:
  - Each pattern element’s duration (like D(2), U(0.5))
    becomes real time in seconds: duration_in_beats * (60 / BPM).
  - Longer beats ring out longer.
  - Quick upstrokes are short and percussive.

  You’ll be asked where to save (for example chordcraft_output.wav).
  Then you can open that WAV in VLC, Windows Media Player, etc.

- Save TAB as TXT
  Saves the currently generated TAB as a .txt file for documentation or sharing.

------------------------------------------------------------
Audio Details
------------------------------------------------------------

- Sample Rate: 48,000 Hz
- Format: 16-bit PCM WAV
- Channels: Mono
- Synthesis: Karplus–Strong (damped feedback loop to mimic a plucked string)
- Strumming feel: simulated by staggering each string a few ms apart (“Strum ms”)

Because we render to a file instead of streaming in real time, this works consistently even on machines / drivers / Bluetooth stacks that crash on live playback.

------------------------------------------------------------
Why this version matters
------------------------------------------------------------

Earlier experiments tried to:
- generate TAB
- play it in real time
- update the GUI cursor live

That live playback can crash Python on some Windows audio drivers (especially certain Bluetooth headsets / USB audio devices) with no Python traceback.

ChordCraft v0.2 fixes that:
- All sound is rendered offline to WAV.
- GUI “playback” is purely visual (silent).
- Timing in the final audio is NOT just uniform steps — it honors your durations per hit, so you actually get groove instead of machine-gun clicks.

This makes it safe to demo in class, during a presentation, or on a different machine.

------------------------------------------------------------
Tested Environment
------------------------------------------------------------

Component / Version:
- OS: Windows 10 / Windows 11
- Python: 3.10+
- NumPy: 1.26+
- Output device: Any WAV player (the program itself does not depend on audio drivers at render time)

Also works on macOS and Linux as long as Tkinter and NumPy are available.

------------------------------------------------------------
Typical demo flow
------------------------------------------------------------

1. Run:
   python chordcraft_gui_groove.py

2. Leave defaults:
   Chords: Em C G D
   Pattern: D(2) D(1) D(0.5) U(0.5)
   BPM: ~70
   Strum ms: ~20

3. Click "Preview TAB" to generate the guitar tab.

4. Click "Scroll Preview (silent)" to show the caret moving.

5. Click "Export WAV (groove)" and save as demo.wav.

6. Double-click the saved WAV file in your OS and play it.

No crashes. No audio driver required. You get audible groove: long beats feel longer, short hits feel snappier.

------------------------------------------------------------
End
------------------------------------------------------------

ChordCraft v0.2 — Interactive Groove Strummer
All rights reserved.
