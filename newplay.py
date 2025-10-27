from pathlib import Path
import os, time, argparse, sys, wave
from typing import List, Tuple
import numpy as np

# ---------- audio config ----------
SAMPLE_RATE = 48000  # match your bluetooth device (locked at 48 kHz)
STRING_BASE = [82.4069, 110.0, 146.832, 195.998, 246.942, 329.628]  # low E..high e
EXPECTED = ['e','B','G','D','A','E']  # we expect lines in this order (top to bottom)

# ---------- tab parsing ----------
def read_text(path:str)->List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.rstrip('\n') for ln in f]

def tokenize(data:str)->List[str]:
    """
    Turn "0--2--2--x--" into tokens ["0","-","-","2","-","-","2","-","-","x","-","-"]
    Digits become whole numbers (e.g. 10, 11), x/X -> 'x', '-' -> '-'
    """
    out = []
    i = 0
    while i < len(data):
        ch = data[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            j = i+1
            while j < len(data) and data[j].isdigit():
                j += 1
            out.append(data[i:j])
            i = j
            continue
        if ch in ['x','X','-']:
            out.append('x' if ch in ['x','X'] else '-')
            i += 1
            continue
        # ignore any other characters silently
        i += 1
    return out

def parse_line(line:str):
    """
    Parse one line like:
    e|--0--2--2--x--
    """
    s = line.strip()
    if not s:
        return None
    if '|' not in s:
        raise ValueError(f"缺少 '|' 分隔符：{s}")
    name, data = s.split('|', 1)
    name = name.strip()
    # normalize string name to one of e,B,G,D,A,E
    name = {
        'e':'e','E':'E',
        'b':'B','B':'B',
        'g':'G','G':'G',
        'd':'D','D':'D',
        'a':'A','A':'A'
    }.get(name, name)

    if name not in EXPECTED:
        raise ValueError(f"未知弦名: {name}，应为 e/B/G/D/A/E")
    tokens = tokenize(data)
    return name, tokens

def parse_tab(lines:List[str]):
    """
    Collect tokens for each string (e,B,G,D,A,E), ensure all are present,
    ensure no duplicates.
    """
    rows = {k: None for k in EXPECTED}
    for ln in lines:
        p = parse_line(ln)
        if not p:
            continue
        name, tokens = p
        if rows[name] is not None:
            raise ValueError(f"重复的弦: {name}")
        rows[name] = tokens
    missing = [k for k,v in rows.items() if v is None]
    if missing:
        raise ValueError(f"缺少弦: {missing}")
    return rows

def pad_rows(rows)->int:
    """
    Make sure all strings have same length by padding '-' at the end
    """
    width = max(len(v) for v in rows.values())
    for k in rows:
        if len(rows[k]) < width:
            rows[k] += ['-'] * (width - len(rows[k]))
    return width

def build_grid(rows)->Tuple[List[str], List[List[str]]]:
    """
    Return (order, grid) where order is ['e','B','G','D','A','E']
    and grid is list of token-lists matching that order.
    """
    order = EXPECTED[:]
    grid = [rows[k] for k in order]
    return order, grid

def load_tab(path:str)->Tuple[List[str], List[List[str]]]:
    lines = read_text(path)
    rows = parse_tab(lines)
    width = pad_rows(rows)
    order, grid = build_grid(rows)
    assert len(order) == 6
    assert all(len(r) == width for r in grid), "各行长度不一致"
    return order, grid

# ---------- pretty print / debug ----------
def segment_for_window(tokens:List[str], L:int, R:int)->str:
    return ''.join(tokens[L:R])

def caret_position(tokens:List[str], L:int, t:int)->int:
    return sum(len(tok) for tok in tokens[L:t])

def render_frame(order, grid, t:int, window:int=16)->str:
    """
    For debugging: show a small sliding window of the tab around step t
    """
    T = len(grid[0])
    L = max(0, t - window//2)
    R = min(T, L + window)
    L = max(0, R - window)
    lines = []
    for name, row in zip(order, grid):
        seg = segment_for_window(row, L, R)
        lines.append(f"{name}|{seg}")
    x = caret_position(grid[0], L, t)
    caret = ' ' * (2 + x) + '^'
    lines.append(caret)
    lines.append(f"[{t+1}/{T}]  window:{L+1}-{R}")
    return '\n'.join(lines)

# ---------- audio synthesis ----------
def string_fret_freq(string_idx:int, fret:int)->float:
    """
    string_idx: 0 = low E string (82.4 Hz), 5 = high e (329 Hz).
    BUT in collect_notes we invert row index, so check that logic.
    """
    return STRING_BASE[string_idx] * (2 ** (fret/12))

def ks_note(freq:float, dur_s:float,
            volume:float=0.22,
            damping:float=0.996,
            pick_brightness:float=0.6)->np.ndarray:
    """
    Karplus-Strong style plucked string
    returns int16 waveform
    """
    N = int(SAMPLE_RATE * dur_s)
    if N <= 0:
        return np.zeros(0, dtype=np.int16)

    delay = max(2, int(round(SAMPLE_RATE / freq)))

    noise = np.random.uniform(-1, 1, delay).astype(np.float32)
    bright = noise - np.concatenate(([0.0], noise[:-1]))
    init = (1 - pick_brightness) * noise + pick_brightness * bright
    init *= 0.9 / max(1.0, np.max(np.abs(init)))

    buf = np.zeros(N, dtype=np.float32)
    y_prev = 0.0
    for n in range(N):
        x = init[n % delay] if n < delay else buf[n - delay]
        y = damping * 0.5 * (x + y_prev)
        buf[n] = y
        y_prev = y

    # mild decay envelope
    t = np.linspace(0, 1, N, dtype=np.float32)
    buf *= (1.0 - 0.25 * t)

    # normalize volume
    peak = np.max(np.abs(buf))
    if peak > 0:
        buf *= (volume / peak)

    return (buf * 32767).astype(np.int16)

def sine_note(freq:float, dur_s:float, volume:float=0.25)->np.ndarray:
    """
    fallback simple sine tone: int16
    """
    t = np.linspace(0, dur_s, int(SAMPLE_RATE*dur_s), endpoint=False)
    wave = np.sin(2*np.pi*freq*t)

    # simple ADSR-ish envelope
    attack = max(1, int(0.01*len(wave)))
    release = max(1, int(0.15*len(wave)))
    env = np.ones_like(wave)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    wave *= env * volume

    return (wave * 32767).astype(np.int16)

def step_interval_sec(bpm:int, steps_per_beat:int)->float:
    """
    how long one 'column' of the tab lasts, in seconds
    """
    return 60.0 / (bpm * steps_per_beat)

def collect_notes(grid:List[List[str]], t:int)->List[Tuple[int,int]]:
    """
    At time-step t, look at each string row,
    if there's a fret number, record (string_index, fret)
    string_index is flipped so that high e -> 5, low E -> 0
    """
    notes = []
    for row_idx, row in enumerate(grid):
        tok = row[t]
        if tok.isdigit():
            fret = int(tok)
            string_idx = 5 - row_idx  # row 0 = 'e' (high), becomes index 5
            notes.append((string_idx, fret))
    return notes

def render_notes_block(notes, dur_s, engine='ks', strum_ms=18, detune_cents_span=6):
    """
    notes: list[(string_idx, fret)]
    return float32 mono audio in [-1,1] for this step
    We simulate a strum by staggering entries with strum_ms between strings.
    """
    if not notes:
        return np.zeros(0, dtype=np.float32)

    waves = []

    # Spread tuning slightly across strings to get a thicker chord sound
    if len(notes) > 1:
        detune = np.linspace(-detune_cents_span/2,
                             detune_cents_span/2,
                             num=len(notes))
    else:
        detune = [0]

    for i, (s, f) in enumerate(notes):
        base_freq = string_fret_freq(s, f)
        adj_freq  = base_freq * (2 ** (detune[i]/1200.0))

        if engine == 'ks':
            raw_int16 = ks_note(adj_freq, dur_s)
        else:
            raw_int16 = sine_note(adj_freq, dur_s)

        raw = raw_int16.astype(np.float32) / 32767.0

        # delay per string to imitate downstroke or upstroke
        delay_samples = int((i * strum_ms / 1000.0) * SAMPLE_RATE)
        if delay_samples > 0:
            raw = np.pad(raw, (delay_samples, 0))

        waves.append(raw)

    # mix down
    L = max(len(w) for w in waves)
    mix = np.zeros(L, dtype=np.float32)
    for w in waves:
        if len(w) < L:
            w = np.pad(w, (0, L-len(w)))
        mix += w

    # normalize
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / (peak * 1.05)

    return mix

def render_full_song_to_array(order, grid,
                              bpm=90,
                              steps_per_beat=2,
                              engine='ks',
                              strum_ms=18):
    """
    stitch every step's block into one long waveform
    """
    dt = step_interval_sec(bpm, steps_per_beat)
    T = len(grid[0])
    pieces = []
    for t in range(T):
        notes = collect_notes(grid, t)
        block = render_notes_block(notes, dt,
                                   engine=engine,
                                   strum_ms=strum_ms)
        pieces.append(block)

    if not pieces:
        song = np.zeros(0, dtype=np.float32)
    else:
        song = np.concatenate(pieces)

    # final gain trim
    peak = np.max(np.abs(song)) if len(song) else 0.0
    if peak > 0:
        song = song / (peak * 1.10)

    return song.astype(np.float32)

def save_wav(path:str, float_wave:np.ndarray):
    """
    float_wave: float32 [-1,1]
    write 16-bit PCM wav
    """
    int16_wave = (float_wave * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)             # mono
        wf.setsampwidth(2)             # 16-bit
        wf.setframerate(SAMPLE_RATE)   # 48 kHz to match your device
        wf.writeframes(int16_wave.tobytes())
    print(f"[OK] wrote {path}")
    if len(int16_wave) > 0:
        length_sec = len(int16_wave) / SAMPLE_RATE
        print(f"length: {length_sec:.2f} seconds")

# ---------- main CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Guitar TAB renderer: parse TXT tab and export a WAV performance (no realtime playback)."
    )
    ap.add_argument("path", help="TAB txt file path")
    ap.add_argument("--bpm", type=int, default=90)
    ap.add_argument("--spb", type=int, default=2, help="steps per beat")
    ap.add_argument("--engine", choices=["ks","sine"], default="ks")
    ap.add_argument("--strum", type=int, default=18,
                    help="strum delay per string in ms, positive = downward sweep")
    ap.add_argument("--out", default="output.wav", help="output wav filename")
    ap.add_argument("--window", type=int, default=24,
                    help="for preview printing only")
    args = ap.parse_args()

    order, grid = load_tab(args.path)
    print(f"Parsed TAB: {len(order)} strings x {len(grid[0])} steps")
    print(render_frame(order, grid, 0, window=args.window))

    song = render_full_song_to_array(
        order, grid,
        bpm=args.bpm,
        steps_per_beat=args.spb,
        engine=args.engine,
        strum_ms=args.strum
    )

    save_wav(args.out, song)
    print("Done. You can now play", args.out, "in any media player.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Demo mode example:")
        print("  python tab_to_wav.py example.txt --bpm 90 --spb 2 --engine ks --out demo.wav")
    else:
        main()
