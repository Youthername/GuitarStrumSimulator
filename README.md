# 🎸 Guitar Strum Simulator 

This project is a standalone **Guitar Strumming Sound Simulator** based on the **Karplus–Strong string synthesis algorithm**.  
It reads 6-line guitar tablature (`.txt`) and generates a corresponding `.wav` audio file that mimics realistic plucked string sounds.

---

## 🧠 Overview

- **Language:** Python 3.8 or higher  
- **Algorithm:** Karplus–Strong plucked string synthesis  
- **Output:** 16-bit PCM `.wav` audio (Mono, 48 kHz)  
- **Mode:** Offline rendering (no real-time playback required)

This version is self-contained and designed for stable, reproducible results — suitable for class projects or audio demos.

---

## 📂 Project Structure

```
GuitarStrumSimulator/
├── newplay.py           # Main script for TAB parsing and audio synthesis
├── example_tab.txt      # Example guitar tab file (C major pattern)
└── README.md            # Project documentation
```

---

## ⚙️ Requirements

| Dependency | Version | Installation |
|-------------|----------|---------------|
| Python | ≥ 3.8 | [Download Python](https://www.python.org/downloads/) |
| NumPy | ≥ 1.21 | `pip install numpy` |

No other external libraries are required.  
**simpleaudio** is not used in this version — playback is done offline by generating `.wav` files.

---

## ▶️ Usage

1. Open a terminal or PowerShell in the project directory.
2. Run the following command:

```bash
python newplay.py example_tab.txt --bpm 90 --spb 2 --engine ks --out my_take.wav
```

### Parameters

| Argument | Description | Default |
|-----------|--------------|----------|
| `path` | Path to your TAB file | *(required)* |
| `--bpm` | Beats per minute | `90` |
| `--spb` | Steps per beat (time resolution) | `2` |
| `--engine` | Synthesis engine: `ks` (Karplus–Strong) or `sine` | `ks` |
| `--strum` | Strumming delay per string in milliseconds (negative = upward) | `18` |
| `--out` | Output `.wav` file name | `my_take.wav` |
| `--window` | Optional: display window width in terminal preview | `24` |

---

## 🪕 Example TAB Format

The simulator expects a **6-line TAB** (EADGBE order).  
Each line should begin with the string name followed by a `|` separator.

```
e|0---0---0---0---
B|1---1---1---1---
G|0---0---0---0---
D|2---2---2---2---
A|3---3---3---3---
E|x---x---x---x---
```

- Numbers: fret positions  
- `x`: muted string  
- `-`: sustain (no new note)

---

## 🎧 Output

After execution, the script generates a `.wav` file in the same directory:

```
[OK] wrote my_take.wav
length: 1.62 seconds
Done. You can now play my_take.wav in any media player.
```

Audio properties:
- Sample Rate: **48000 Hz**  
- Bit Depth: **16-bit PCM**  
- Channel: **Mono**  
- Duration: depends on TAB length and BPM  

This sampling rate ensures compatibility with most Bluetooth and wired devices.

---

## 🧩 Advanced Options

You can experiment with parameters:

```bash
# Slower tempo, sine wave engine
python newplay.py example_tab.txt --bpm 70 --engine sine --out mellow_take.wav

# Faster tempo, reverse strumming direction
python newplay.py example_tab.txt --bpm 120 --strum -15 --out fast_upstroke.wav
```

---

## 💡 Notes

- The program is fully offline; it **does not depend on your audio device**.  
- Works on **Windows, macOS, and Linux**.  
- You can use any Python IDE or terminal to run the script.  
- If you want to develop a GUI or real-time version later, this serves as a **stable baseline**.

---

## 🧪 Tested Environment

| Component | Version |
|------------|----------|
| OS | Windows 10 / 11 |
| Python | 3.10.14 |
| NumPy | 1.26.4 |
| Audio Output | 48 kHz Bluetooth / Realtek Speaker |

---

## 🏁 Author

**Shuo Li**  
University of Iowa · Department of Statistics and Actuarial Science  
📧 [li-shuo@uiowa.edu](mailto:li-shuo@uiowa.edu)

---

© 2025 Shuo Li — *Guitar Strum Simulator Stable Version v1.0*
