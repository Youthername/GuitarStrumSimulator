# chordcraft_gui_groove.py
# 扫弦模拟器 / ChordCraft v0.2
#
# - GUI 输入和弦 / 节奏型 / BPM
# - 预览谱面：生成TAB文本供查看
# - 预览滚动(静音)：用旧的均匀格子逻辑滚动光标（视觉演示，不出声）
# - 导出WAV：使用“有节奏时值”的新版合成器
#             -> 你的 D(1) / U(0.5) / D(0.25) 真正决定时长和律动
#
# 依赖:
#   pip install numpy
#
# 不调用 simpleaudio，绝对稳定，不会卡死。

import time, threading, wave
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ------------------ 常量 & 和弦库 ------------------

SAMPLE_RATE = 48000

# 物理弦频率 (标准调弦): 低音E(0) -> 高音e(5)
STRING_BASE = [82.4069, 110.0, 146.832, 195.998, 246.942, 329.628]

# TAB里显示的顺序（从高音弦到低音弦）
EXPECTED = ['e','B','G','D','A','E']

# CHORDS: [E,A,D,G,B,e]  低E→高e
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

# ------------------ 节奏解析、TAB生成 ------------------

def chord_to_tokens(shape: List[int]) -> List[str]:
    """
    输入和弦按法: [E,A,D,G,B,e]  (-1 = 不弹)
    输出TAB行对应token: ['e','B','G','D','A','E'] 顺序
    """
    E,A,D,G,B,e = shape
    def enc(f): return 'x' if f < 0 else str(f)
    return [enc(e), enc(B), enc(G), enc(D), enc(A), enc(E)]

def parse_pattern(pat_str: str) -> List[Tuple[str,float]]:
    """
    "D(1) U(0.5) D(0.25)" -> [("D",1.0),("U",0.5),("D",0.25)]
    也支持 D:1 D1 这种写法
    """
    import re
    ST = re.compile(r"(?i)\b([DU])\s*(?:\(|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*\)?")
    out=[]
    for m in ST.finditer(pat_str.replace(',', ' ')):
        out.append((m.group(1).upper(), float(m.group(2))))
    if not out:
        raise ValueError("节奏格式不认识，例如：D(1) U(0.5)")
    return out

def generate_tab_text(
    chord_names: List[str],
    pattern: List[Tuple[str,float]],
    spb: int,
    cycles: int = 1,
    gap_char: str='-'
) -> str:
    """
    旧逻辑: 为了画TAB，我们依然用spb把时值量化成格子。
    注意：这只是视觉TAB用，真正的音频将不会用这个量化。
    """
    lines = {k: [] for k in EXPECTED}  # e,B,G,D,A,E
    for _ in range(cycles):
        for name in chord_names:
            if name not in CHORDS:
                raise KeyError(f"未知和弦：{name}")
            hit_tokens = chord_to_tokens(CHORDS[name])
            for _, dur in pattern:
                steps = max(1, round(dur * spb))
                # 首格：击弦
                for k, tok in zip(EXPECTED, hit_tokens):
                    lines[k].append(tok)
                # 剩下格：延音
                for __ in range(steps-1):
                    for k in EXPECTED:
                        lines[k].append(gap_char)

    return "\n".join([f"{k}|{''.join(lines[k])}" for k in EXPECTED])

# ------------------ TAB解析成网格 (给滚动预览用) ------------------

def normalize_tab_line_data(data: str) -> str:
    norm_map = {'—':'-','–':'-','－':'-','\xa0':' '}
    for a,b in norm_map.items():
        data = data.replace(a,b)
    return data

def tokenize_line_to_columns(data: str) -> List[str]:
    """
    "10--8-x-" -> ["10","-","-","8","-","x","-"]
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
    返回:
      order=['e','B','G','D','A','E']
      grid = [ row_for_e, row_for_B, ... ] (定长列数组)
    """
    rows = {k: None for k in EXPECTED}
    for raw in tab_text.splitlines():
        if not raw.strip(): continue
        if '|' not in raw: continue
        left,right = raw.split('|',1)
        name = left.strip()
        name = {'e':'e','E':'E','b':'B','B':'B','g':'G','G':'G','d':'D','D':'D','a':'A','A':'A'}.get(name,name)
        if name not in rows: continue
        rows[name] = tokenize_line_to_columns(right)

    missing = [k for k,v in rows.items() if v is None]
    if missing:
        raise ValueError(f"缺少弦：{missing} (需要 e| B| G| D| A| E| )")

    width = max(len(v) for v in rows.values())
    for k in rows:
        if len(rows[k])<width:
            rows[k] += ['-']*(width-len(rows[k]))

    order = EXPECTED[:]
    grid = [rows[k] for k in order]
    return order, grid

# ------------------ GUI滚动帧渲染 (视觉预览用) ------------------

def render_frame(order, grid, t:int, window:int=24) -> str:
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

    caret_pad = 2 + cursor_x  # "e|"是2列
    if caret_pad<0: caret_pad=0
    lines.append(' '*caret_pad + '^')
    lines.append(f"[{t+1}/{T}] window:{L+1}-{R}")
    return "\n".join(lines)

# ------------------ 声音合成部件 ------------------

def tabrow_index_to_physical_string_idx(row_idx: int) -> int:
    """
    我们显示顺序: e,B,G,D,A,E (row_idx=0..5)
    物理弦顺序:   E,A,D,G,B,e (index=0..5)
    对应: phys_idx = 5-row_idx
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
    Karplus-Strong 拟拨弦
    返回 int16 mono
    """
    N = int(SAMPLE_RATE*dur)
    if N<=0:
        return np.zeros(0,np.int16)

    delay = max(2, int(round(SAMPLE_RATE/freq)))
    noise = np.random.uniform(-1,1,delay).astype(np.float32)

    # pick_brightness: 明亮击弦
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

    # 线性收尾，避免无限延音
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
    对一个和弦的一次扫弦，合成长度=hit_duration_sec的一段音频。
    - chord_shape 是 [E,A,D,G,B,e] 的品位
    - strum_ms 控制弦与弦之间的扫弦延迟
    """
    # 我们要把和弦shape映射到 e,B,G,D,A,E 的顺序，
    # 然后从高音弦往低音弦扫（高->低听起来像往下刷）。
    # chord_to_tokens给的是可视token，包含 'x' 和数字字符串，
    # 我们要拿回真实 (phys_idx, fret) 列表。
    eTok,BTok,GTok,DTok,ATok,ETok = chord_to_tokens(chord_shape)

    # 可视行 index -> 物理弦index
    # row_idx 0=e -> phys=5, row_idx5=E -> phys=0
    vis_tokens = [eTok,BTok,GTok,DTok,ATok,ETok]

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

    # 逐弦合成并叠加，加入扫弦延迟
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

# ------------------ 新：按真实时值生成最终音频 ------------------

def synthesize_song_with_durations(
    chords_list: List[str],
    pattern: List[Tuple[str, float]],
    bpm: int,
    cycles: int,
    strum_ms: int
) -> np.ndarray:
    """
    真正有律动的版本。
    对每个和弦，按顺序依次扫 pattern 中的每一下。
    每一下的实际持续时间 = (dur_in_pattern_beats * 60 / bpm) 秒
    把每一下变成音频块，然后直接串接。
    """
    beat_sec = 60.0 / bpm  # 一拍多长(秒)

    chunks: List[np.ndarray] = []

    for _ in range(cycles):
        for chord_name in chords_list:
            if chord_name not in CHORDS:
                raise KeyError(f"未知和弦：{chord_name}")
            chord_shape = CHORDS[chord_name]

            for _, dur_beats in pattern:
                hit_len = dur_beats * beat_sec  # 这一击实际持续时间
                block = mix_chord_hit(
                    chord_shape=chord_shape,
                    hit_duration_sec=hit_len,
                    strum_ms=strum_ms
                )
                chunks.append(block.astype(np.int16))

    if not chunks:
        return np.zeros(0, np.int16)

    full = np.concatenate(chunks)

    # 最后做一个轻微限幅，防止过载
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

# ------------------ 旧滚动用的合成 (只是为了滚动预览动画) ------------------

def synthesize_for_scroll_preview(tab_text: str, bpm:int, spb:int, strum_ms:int):
    """
    还是老逻辑：把TAB摊成grid，用固定dt = 60/(bpm*spb)
    返回 order,grid,T,dt
    (注意 dt 在这里是“每一格固定时长”，仅用于视觉滚动)
    """
    order, grid = parse_tab_text(tab_text)
    T = len(grid[0])
    dt = 60.0/(bpm*spb)  # 均匀格长 (视觉模拟用)
    return order, grid, T, dt

# ------------------ GUI 应用 ------------------

@dataclass
class Settings:
    chords: str = "Em C G D"
    pattern: str = "D(1) D(0.5) D(0.25) U(0.25)"
    bpm: int = 70       # 稍慢一点，便于听到律动
    spb: int = 4        # 仅用于TAB视图细分 & 视觉滚动
    cycles: int = 2
    window: int = 24
    strum: int = 20     # 扫弦延迟ms，稍大一点更像人手

class App:
    def __init__(self, root):
        self.root = root
        root.title("扫弦模拟器 / ChordCraft v0.2 (有律动版)")
        self.set = Settings()
        self._stop = False

        frm = ttk.Frame(root, padding=10)
        frm.pack(fill='both', expand=True)

        def row(label, widget):
            r = ttk.Frame(frm)
            r.pack(fill='x', pady=4)
            ttk.Label(r, text=label, width=10).pack(side='left')
            widget.pack(side='left', fill='x', expand=True)
            return r

        # 和弦序列
        self.ent_chords = ttk.Entry(frm)
        self.ent_chords.insert(0, self.set.chords)
        row("和弦序列", self.ent_chords)

        # 节奏型
        self.ent_pat = ttk.Entry(frm)
        self.ent_pat.insert(0, self.set.pattern)
        row("节奏型", self.ent_pat)

        # 数值区域
        grid2 = ttk.Frame(frm)
        grid2.pack(fill='x', pady=4)
        for label, var, w in [
            ("BPM", 'bpm', 8),
            ("SPB", 'spb', 8),
            ("循环", 'cycles', 8),
            ("窗口", 'window', 8),
            ("扫弦ms", 'strum', 8),
        ]:
            f = ttk.Frame(grid2)
            f.pack(side='left', padx=6)
            ttk.Label(f, text=label).pack()
            e = ttk.Entry(f, width=w)
            e.insert(0, getattr(self.set, var))
            setattr(self, f"ent_{var}", e)
            e.pack()

        # 按钮区
        btns = ttk.Frame(frm)
        btns.pack(fill='x', pady=6)

        ttk.Button(btns, text="预览谱面", command=self.preview_tab).pack(side='left', padx=4)
        ttk.Button(btns, text="预览滚动(静音)", command=self.preview_scroll).pack(side='left', padx=4)
        ttk.Button(btns, text="停止滚动", command=self.stop_scroll).pack(side='left', padx=4)
        ttk.Button(btns, text="导出WAV(有节奏)", command=self.export_wav).pack(side='left', padx=4)
        ttk.Button(btns, text="保存为TXT", command=self.save_txt).pack(side='left', padx=4)

        # 文本框输出区
        self.txt = tk.Text(frm, height=20, font=("Consolas", 11))
        self.txt.pack(fill='both', expand=True)
        self.txt.insert('end',
            "欢迎使用 ChordCraft v0.2 (有律动)\n"
            "1. 填和弦和节奏 (D(1) U(0.5) ...)\n"
            "2. 预览谱面 看六线谱\n"
            "3. 预览滚动(静音) 看指针走位\n"
            "4. 导出WAV(有节奏) 听真实扫弦，强拍长、弱拍短\n"
        )

    # ---- 内部工具 ----
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
            messagebox.showerror("参数错误", str(e))
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

    # ---- 按钮回调 ----
    def preview_tab(self):
        try:
            tab_txt = self.gen_tab_text()
        except Exception:
            return
        self.txt.delete('1.0','end')
        self.txt.insert('end', tab_txt)

    def preview_scroll(self):
        """
        静音播放指针滚动：
        这里用旧的均匀dt逻辑，只是视觉化，不代表真实时值。
        但它不会卡死、不会碰声卡。
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
            messagebox.showerror("预览失败", str(e))
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
        导出真正有律动的音频：
        - 用新的 synthesize_song_with_durations
        - D(1) vs D(0.25) 真正时值不同
        """
        try:
            s = self.get_settings()
            pat_list = parse_pattern(s.pattern)
            chords_list = s.chords.split()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
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
            messagebox.showerror("导出失败", str(e))
            return

        messagebox.showinfo(
            "导出完成",
            f"已生成带节奏律动的WAV:\n{p}\n用系统播放器打开就能听。"
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
        messagebox.showinfo("已保存", p)

# ------------------ main ------------------

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
