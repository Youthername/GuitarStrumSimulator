# chordcraft_gui_groove.py
# ChordCraft v0.2 - Interactive Guitar Strum Synth
#
# Features:
# - GUI input for chord progression / strumming pattern / tempo
# - "Preview TAB": generate a 6-line guitar TAB for the current settings
# - "Scroll Preview (silent)": animated cursor scrolling through the TAB
#   using a uniform time grid (visual only, no audio playback, no crash risk)
# - "Export WAV (groove)": renders actual audio with realistic timing,
#   where D(2), D(1), D(0.5), U(0.5) durations are respected
#
# Tech:
# - Offline synthesis using Karplus–Strong string model
# - No real-time audio playback (simpleaudio not required)
# - Output is a 16-bit mono WAV at 48 kHz
#
# Dependencies:
#   pip install numpy
#
# This version is designed to be stable on Windows/macOS/Linux.

import time, threading, wave
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ------------------ Constants & Chord Shapes ------------------

SAMPLE_RATE = 48000

# Physical string base frequencies in standard tuning:
# Low E (index 0) -> A -> D -> G -> B -> high e (index 5)
STRING_BASE = [82.4069, 110.0, 146.832, 195.998, 246.942, 329.628]

# TAB display order (top to bottom)
EXPECTED = ['e','B','G','D','A','E']

# Chord dictionary:
# Each chord is described as fret numbers for [E, A, D, G, B, e]
# -1 means "do not play / muted"
CHORDS: Dict[str, List[int]] = {
    "C":      [-1, 3, 2, 0, 1, 0],
    "D":      [-1, -1, 0, 2, 3, 2],
    "E":      [0, 2, 2, 1, 0, 0],
    "F":      [1, 3, 3, 2, 1, 1],
    "G":      [3, 2, 0, 0, 0, 3],
    "A":      [-1, 0, 2, 2, 2, 0],
    "Am":     [-1, 0, 2, 2, 1, 0],
    "Dm":     [-1, -1, 0, 2, 3, 1],
    "Em":     [0, 2, 2, 0, 0, 0],
    "Bm":     [2, 2, 4, 4, 3, 2],
    "A7":     [-1, 0, 2, 0, 2, 0],
    "E7":     [0, 2, 0, 1, 0, 0],
    "G7":     [3, 2, 0, 0, 0, 1],
    "Cadd9":  [-1, 3, 2, 0, 3, 3],
    "Gsus4":  [3, 2, 0, 0, 1, 3],
    "Dsus4":  [-1, -1, 0, 2, 3, 3],
    "Dsus2":  [-1, -1, 0, 2, 3, 0],
    "Asus2":  [-1, 0, 2, 2, 0, 0],
    "Asus4":  [-1, 0, 2, 2, 3, 0],
}

# ------------------ Pattern parsing / TAB generation ------------------

def chord_to_tokens(shape: List[int]) -> List[str]:
    """
    Input chord fingering as [E,A,D,G,B,e] (low E -> high e).
    Output tokens in TAB order [e,B,G,D,A,E].
    -1 becomes 'x' (muted), otherwise it's fret number.
    """
    E,A,D,G,B,e = shape
    def enc(f): return 'x' if f < 0 else str(f)
    return [enc(e), enc(B), enc(G), enc(D), enc(A), enc(E)]

def parse_pattern(pat_str: str) -> List[Tuple[str,float]]:
    """
    Parse a human strumming pattern string like:
        "D(2) D(1) D(0.5) U(0.5)"
    or:
        "D:2 D1 U0.5"
    Returns:
        [("D", 2.0), ("D", 1.0), ("D", 0.5), ("U", 0.5)]
    Direction 'D' or 'U' is captured but currently we just treat them
    as timing events (not different synthesis yet).
    """
    import re
    ST = re.compile(r"(?i)\b([DU])\s*(?:\(|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*\)?")
    out=[]
    for m in ST.finditer(pat_str.replace(',', ' ')):
        out.append((m.group(1).upper(), float(m.group(2))))
    if not out:
        raise ValueError("Pattern parse failed. Try e.g. D(1) U(0.5).")
    return out

def generate_tab_text(
    chord_names: List[str],
    pattern: List[Tuple[str,float]],
    spb: int,
    cycles: int = 1,
    gap_char: str='-'
) -> str:
    """
    This produces a visual TAB (text only).
    We still quantize durations using 'spb' (steps per beat)
    just for drawing. This does NOT affect audio anymore.
    """
    lines = {k: [] for k in EXPECTED}  # e,B,G,D,A,E
    for _ in range(cycles):
        for name in chord_names:
            if name not in CHORDS:
                raise KeyError(f"Unknown chord: {name}")
            hit_tokens = chord_to_tokens(CHORDS[name])
            for _, dur in pattern:
                steps = max(1, round(dur * spb))
                # first column: strike
                for k, tok in zip(EXPECTED, hit_tokens):
                    lines[k].append(tok)
                # sustain columns:
                for __ in range(steps-1):
                    for k in EXPECTED:
                        lines[k].append(gap_char)

    return "\n".join([f"{k}|{''.join(lines[k])}" for k in EXPECTED])

# ------------------ TAB parsing for scroll preview ------------------

def normalize_tab_line_data(data: str) -> str:
    norm_map = {'—':'-','–':'-','－':'-','\xa0':' '}
    for a,b in norm_map.items():
        data = data.replace(a,b)
    return data

def tokenize_line_to_columns(data: str) -> List[str]:
    """
    Convert a string like "10--8-x-" into ["10","-","-","8","-","x","-"]
    Handles multi-digit frets.
    """
    data = normalize_tab_line_data(data)
    out=[]; i=0
    while i<len(data):
        ch=data[i]
        if ch.isdigit():
            j=i+1
            while j<len(data) and data[j].isdigit():
                j+=1
            out.append(data[i:j])
            i=j
            continue
        if ch in 'xX':
            out.append('x'); i+=1; continue
        if ch=='-':
            out.append('-'); i+=1; continue
        i+=1
    return out

def parse_tab_text(tab_text: str):
    """
    Returns:
      order ['e','B','G','D','A','E']
      grid  [row_e, row_B, ..., row_E]
    Each row_* is a list of column tokens, all padded to equal length.
    This is used only for the visual "Scroll Preview (silent)".
    """
    rows = {k: None for k in EXPECTED}
    for raw in tab_text.splitlines():
        if not raw.strip(): continue
        if '|' not in raw: continue
        left,right = raw.split('|',1)
        name = left.strip()
        # normalize string labels
        name = {'e':'e','E':'E','b':'B','B':'B','g':'G','G':'G','d':'D','D':'D','a':'A','A':'A'}.get(name,name)
        if name not in rows:
            continue
        rows[name] = tokenize_line_to_columns(right)

    missing = [k for k,v in rows.items() if v is None]
    if missing:
        raise ValueError(f"Missing strings in TAB: {missing} (need e,B,G,D,A,E)")

    width = max(len(v) for v in rows.values())
    for k in rows:
        if len(rows[k])<width:
            rows[k] += ['-']*(width-len(rows[k]))

    order = EXPECTED[:]
    grid = [rows[k] for k in order]
    return order, grid

# ------------------ Scroll-frame renderer (visual only) ------------------

def render_frame(order, grid, t:int, window:int=24) -> str:
    """
    Render a moving-window view of the TAB with a caret "^"
    under the current column t.
    This is purely visual timing (uniform dt).
    """
    T = len(grid[0])
    if T==0:
        return "(empty)"
    if t<0: t=0
    if t>=T: t=T-1

    L = max(0, min(t - window//2, T-window))
    R = min(T, L+window)

    cursor_x = sum(len(tok) for tok in grid[0][L:t])
    if cursor_x < 0: cursor_x = 0

    lines=[]
    for nm,row in zip(order,grid):
        seg=''.join(row[L:R])
        lines.append(f"{nm}|{seg}")

    caret_pad = 2 + cursor_x  # "e|" takes 2 chars
    if caret_pad<0: caret_pad=0
    lines.append(' '*caret_pad + '^')
    lines.append(f"[{t+1}/{T}] window:{L+1}-{R}")
    return "\n".join(lines)

# ------------------ Low-level synthesis helpers ------------------

def tabrow_index_to_physical_string_idx(row_idx: int) -> int:
    """
    Visual TAB order is [e,B,G,D,A,E] (row_idx=0..5).
    Physical tuning order (STRING_BASE) is [E,A,D,G,B,e] (index=0..5).
    Mapping is phys_idx = 5 - row_idx.
    """
    return 5 - row_idx

def string_fret_freq(phys_idx: int, fret: int) -> Optional[float]:
    if fret < 0: return None
    if fret > 30: return None
    base = STRING_BASE[phys_idx]
    return base * (2 ** (fret/12.0))

def ks_note(freq: float, dur: float,
            volume=0.22,
            damping=0.996,
            pick_brightness=0.6) -> np.ndarray:
    """
    Karplus–Strong plucked string synthesis.
    Returns int16 mono.
    """
    N = int(SAMPLE_RATE*dur)
    if N<=0:
        return np.zeros(0,np.int16)

    delay = max(2, int(round(SAMPLE_RATE/freq)))
    noise = np.random.uniform(-1,1,delay).astype(np.float32)

    # brighten initial excitation
    bright = noise - np.concatenate(([0.0], noise[:-1]))
    init = (1-pick_brightness)*noise + pick_brightness*bright
    peak = np.max(np.abs(init))
    if peak>0:
        init *= (0.9/peak)

    buf = np.zeros(N, np.float32)
    y=0.0
    for n in range(N):
        x = init[n%delay] if n<delay else buf[n-delay]
        y = damping*0.5*(x+y)
        buf[n]=y

    # simple decay envelope
    t_arr = np.linspace(0,1,N, np.float32)
    buf *= (1-0.25*t_arr)

    peak2 = np.max(np.abs(buf))
    if peak2>0:
        buf *= (volume/peak2)

    return (buf*32767).astype(np.int16)

def mix_chord_hit(
    chord_shape: List[int],
    hit_duration_sec: float,
    strum_ms: int
) -> np.ndarray:
    """
    Synthesize one strum of a chord over hit_duration_sec.
    The chord is the shape [E,A,D,G,B,e].
    strum_ms controls per-string delay (gives human-like sweep).
    """

    # Convert chord fingering to tokens in TAB order [e,B,G,D,A,E]
    eTok,BTok,GTok,DTok,ATok,ETok = chord_to_tokens(chord_shape)
    vis_tokens = [eTok,BTok,GTok,DTok,ATok,ETok]

    # Collect playable notes as (physical_string_index, fret)
    notes = []
    for row_idx, tok in enumerate(vis_tokens):
        if tok in ('x','X','-'):
            continue
        try:
            fret_val = int(tok)
        except:
            continue
        phys_idx = tabrow_index_to_physical_string_idx(row_idx)
        freq0 = string_fret_freq(phys_idx, fret_val)
        if freq0 is None:
            continue
        notes.append((phys_idx, fret_val))

    if not notes:
        return np.zeros(int(SAMPLE_RATE*hit_duration_sec), np.int16)

    # Build each string's waveform, offset in time to simulate strum
    waves=[]
    for i,(phys_idx, fret_val) in enumerate(notes):
        freq0 = string_fret_freq(phys_idx, fret_val)
        if freq0 is None:
            continue
        w = ks_note(freq0, hit_duration_sec)

        delay_samples = int((i * (strum_ms/1000.0))*SAMPLE_RATE)
        if delay_samples>0:
            w = np.pad(w,(delay_samples,0))

        waves.append(w.astype(np.float32)/32767.0)

    if not waves:
        return np.zeros(int(SAMPLE_RATE*hit_duration_sec), np.int16)

    # Mix down
    L = max(len(w) for w in waves)
    mix = np.zeros(L, np.float32)
    for w in waves:
        if len(w)<L:
            w = np.pad(w,(0,L-len(w)))
        mix += w

    peak = float(np.max(np.abs(mix)))
    if peak>0:
        mix /= (peak*1.05)

    return (mix*32767).astype(np.int16)

# ------------------ Groove-aware synthesis ------------------

def synthesize_song_with_durations(
    chords_list: List[str],
    pattern: List[Tuple[str, float]],
    bpm: int,
    cycles: int,
    strum_ms: int
) -> np.ndarray:
    """
    This is the "groove" renderer.
    We no longer assume uniform fixed steps.
    Instead, each pattern entry has its own duration in beats.
    beat_sec = 60 / bpm.
    For every chord in the progression, for every pattern element,
    we generate one strum whose length = dur_beats * beat_sec.
    We then concatenate those audio chunks in order.
    """
    beat_sec = 60.0 / bpm
    chunks: List[np.ndarray] = []

    for _ in range(cycles):
        for chord_name in chords_list:
            if chord_name not in CHORDS:
                raise KeyError(f"Unknown chord: {chord_name}")
            chord_shape = CHORDS[chord_name]

            for _, dur_beats in pattern:
                hit_len = dur_beats * beat_sec
                block = mix_chord_hit(
                    chord_shape=chord_shape,
                    hit_duration_sec=hit_len,
                    strum_ms=strum_ms
                )
                chunks.append(block.astype(np.int16))

    if not chunks:
        return np.zeros(0, np.int16)

    full = np.concatenate(chunks)

    # final limiter-ish normalization
    peak = np.max(np.abs(full))
    if peak>0:
        full = (full.astype(np.float32)/(peak*1.05)*32767).astype(np.int16)

    return full

def save_wav(path: str, audio_int16: np.ndarray):
    with wave.open(path,'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

# ------------------ Preview-only synthesis for scroll animation ------------------

def synthesize_for_scroll_preview(tab_text: str, bpm:int, spb:int, strum_ms:int):
    """
    This preserves the legacy "grid timing" model:
    - parse_tab_text() builds a fixed grid of columns
    - dt = 60 / (bpm * spb) is constant between columns
    We use this ONLY to animate the caret in Scroll Preview.
    """
    order, grid = parse_tab_text(tab_text)
    T = len(grid[0])
    dt = 60.0/(bpm*spb)
    return order, grid, T, dt

# ------------------ GUI ------------------

@dataclass
class Settings:
    chords: str = "Em C G D"
    pattern: str = "D(2) D(1) D(0.5) U(0.5)"
    bpm: int = 70        # tempo for groove synth
    spb: int = 2         # steps per beat (for TAB drawing & scroll preview)
    cycles: int = 2
    window: int = 24
    strum: int = 20      # ms between strings in a strum

class App:
    def __init__(self, root):
        self.root = root
        root.title("ChordCraft v0.2 - Interactive Groove Strummer")
        self.set = Settings()
        self._stop = False

        frm = ttk.Frame(root, padding=10)
        frm.pack(fill='both', expand=True)

        def row(label, widget):
            r = ttk.Frame(frm)
            r.pack(fill='x', pady=4)
            ttk.Label(r, text=label, width=14).pack(side='left')
            widget.pack(side='left', fill='x', expand=True)
            return r

        # Chord progression
        self.ent_chords = ttk.Entry(frm)
        self.ent_chords.insert(0, self.set.chords)
        row("Chords", self.ent_chords)

        # Strumming pattern
        self.ent_pat = ttk.Entry(frm)
        self.ent_pat.insert(0, self.set.pattern)
        row("Pattern", self.ent_pat)

        # Numeric panel
        grid2 = ttk.Frame(frm)
        grid2.pack(fill='x', pady=4)
        for label, var, w in [
            ("BPM", 'bpm', 8),
            ("SPB", 'spb', 8),
            ("Loops", 'cycles', 8),
            ("Window", 'window', 8),
            ("Strum ms", 'strum', 8),
        ]:
            f = ttk.Frame(grid2)
            f.pack(side='left', padx=6)
            ttk.Label(f, text=label).pack()
            e = ttk.Entry(f, width=w)
            e.insert(0, getattr(self.set, var))
            setattr(self, f"ent_{var}", e)
            e.pack()

        # Buttons
        btns = ttk.Frame(frm)
        btns.pack(fill='x', pady=6)

        ttk.Button(btns, text="Preview TAB", command=self.preview_tab).pack(side='left', padx=4)
        ttk.Button(btns, text="Scroll Preview (silent)", command=self.preview_scroll).pack(side='left', padx=4)
        ttk.Button(btns, text="Stop Scroll", command=self.stop_scroll).pack(side='left', padx=4)
        ttk.Button(btns, text="Export WAV (groove)", command=self.export_wav).pack(side='left', padx=4)
        ttk.Button(btns, text="Save TAB as TXT", command=self.save_txt).pack(side='left', padx=4)

        # Output text area
        self.txt = tk.Text(frm, height=20, font=("Consolas", 11))
        self.txt.pack(fill='both', expand=True)
        self.txt.insert('end',
            "ChordCraft v0.2 (Groove Edition)\n"
            "1. Enter chord progression (e.g. Em C G D)\n"
            "2. Enter pattern with durations (e.g. D(2) D(1) D(0.5) U(0.5))\n"
            "3. Preview TAB to see the generated 6-line guitar tab\n"
            "4. Scroll Preview (silent) to visualize timing cursor\n"
            "5. Export WAV (groove) to render realistic strumming audio\n"
            "   - Long beats last longer (D(2)), short hits are quick (U(0.5))\n"
        )

    # ---- helpers ----
    def get_settings(self) -> Settings:
        try:
            return Settings(
                chords = self.ent_chords.get().strip(),
                pattern= self.ent_pat.get().strip(),
                bpm    = int(self.ent_bpm.get()),
                spb    = int(self.ent_spb.get()),
                cycles = int(self.ent_cycles.get()),
                window = int(self.ent_window.get()),
                strum  = int(self.ent_strum.get()),
            )
        except Exception as e:
            messagebox.showerror("Invalid parameter", str(e))
            raise

    def gen_tab_text(self) -> str:
        s = self.get_settings()
        pat_list = parse_pattern(s.pattern)
        chord_list = s.chords.split()
        return generate_tab_text(chord_list, pat_list, s.spb, s.cycles)

    def _set_text_async(self, content: str):
        def do():
            self.txt.delete('1.0','end')
            self.txt.insert('end', content)
        self.txt.after(0, do)

    # ---- button callbacks ----
    def preview_tab(self):
        try:
            tab_txt = self.gen_tab_text()
        except Exception:
            return
        self.txt.delete('1.0','end')
        self.txt.insert('end', tab_txt)

    def preview_scroll(self):
        """
        Animate caret over TAB columns using uniform dt timing.
        This does not reflect groove durations,
        but it's safe, silent, and visually easy to explain.
        """
        try:
            tab_txt = self.gen_tab_text()
            s = self.get_settings()
        except Exception:
            return

        try:
            order, grid, T, dt = synthesize_for_scroll_preview(
                tab_text = tab_txt,
                bpm      = s.bpm,
                spb      = s.spb,
                strum_ms = s.strum
            )
        except Exception as e:
            messagebox.showerror("Preview failed", str(e))
            return

        self._stop = False

        def runner():
            start = time.perf_counter()
            for t in range(T):
                if self._stop:
                    break
                frame_txt = render_frame(order, grid, t, window=s.window)
                self._set_text_async(frame_txt)

                target = start + (t+1)*dt
                now = time.perf_counter()
                delay = target - now
                if delay>0:
                    time.sleep(delay)

        threading.Thread(target=runner, daemon=True).start()

    def stop_scroll(self):
        self._stop = True

    def export_wav(self):
        """
        Render groove-aware audio and write it to a WAV file.
        Each pattern element's duration (like D(2), U(0.5))
        becomes actual time in the output.
        """
        try:
            s = self.get_settings()
            pat_list = parse_pattern(s.pattern)
            chords_list = s.chords.split()
        except Exception as e:
            messagebox.showerror("Invalid parameter", str(e))
            return

        p = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV audio","*.wav")],
            initialfile="chordcraft_output.wav"
        )
        if not p:
            return

        try:
            audio_full = synthesize_song_with_durations(
                chords_list = chords_list,
                pattern     = pat_list,
                bpm         = s.bpm,
                cycles      = s.cycles,
                strum_ms    = s.strum
            )
            save_wav(p, audio_full)
        except Exception as e:
            messagebox.showerror("Export failed", str(e))
            return

        messagebox.showinfo(
            "Export complete",
            f"WAV with groove timing written:\n{p}\n"
            "Open it in any media player."
        )

    def save_txt(self):
        try:
            tab_txt = self.gen_tab_text()
        except Exception:
            return

        p = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text","*.txt")],
            initialfile="generated_tab.txt"
        )
        if not p:
            return

        Path(p).write_text(tab_txt, encoding='utf-8')
        messagebox.showinfo("Saved", p)

# ------------------ main ------------------

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
