import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from extract_wav_features import classify_audio, get_filter_suggestion
import soundfile as sf
import numpy as np
from filter import process_audio
from plot_audio_comparison import plot_comparison
import threading
import tempfile
import wave
import pyaudio

# 实时处理模块导入
try:
    from realtime_audio_processing import RealTimeAudioProcessor
    HAS_REALTIME = True
except ImportError:
    HAS_REALTIME = False

class AudioApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("音频噪声分类与滤波系统")
        self.geometry("900x700")
        self.configure(bg='#f4f4f4')
        self.resizable(True, True)  # 允许窗口调节大小

        # 中文友好字体
        cn_font = ('微软雅黑', 12)
        cn_font_bold = ('微软雅黑', 13, 'bold')
        cn_font_text = ('微软雅黑', 11)

        # 样式
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=cn_font, padding=6)
        style.configure('TLabel', font=cn_font, background='#f4f4f4')
        style.configure('TFrame', background='#f4f4f4')
        style.configure('TLabelframe.Label', font=cn_font_bold)
        style.configure('TLabelframe', background='#f4f4f4')

        # 文件选择区
        file_frame = ttk.Frame(self)
        file_frame.pack(fill=tk.X, pady=10, padx=10)
        self.file_label = ttk.Label(file_frame, text="音频文件:")
        self.file_label.pack(side=tk.LEFT)
        self.file_entry = ttk.Entry(file_frame, width=60, font=cn_font)
        self.file_entry.pack(side=tk.LEFT, padx=5)
        self.browse_btn = ttk.Button(file_frame, text="选择文件", command=self.browse_file)
        self.browse_btn.pack(side=tk.LEFT, padx=5)

        # 拖拽区域
        self.drop_label = tk.Label(self, text="或将音频文件拖拽到此处", bg='#e0e0e0', fg='black',
                                   font=cn_font_bold, relief="groove", bd=2, width=60, height=2)
        self.drop_label.pack(pady=5)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.drop)

        # 参数设置区
        param_frame = ttk.LabelFrame(self, text="滤波参数设置")
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(param_frame, text="滤波类型:").grid(row=0, column=0, padx=5, pady=2)
        self.filter_type_var = tk.StringVar(value="auto")
        ttk.Combobox(param_frame, textvariable=self.filter_type_var, values=["auto", "iir", "lms"], width=8, state="readonly", font=cn_font).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="陷波频率(f0):").grid(row=0, column=2, padx=5)
        self.f0_var = tk.DoubleVar(value=50.0)
        ttk.Entry(param_frame, textvariable=self.f0_var, width=8, font=cn_font).grid(row=0, column=3, padx=5)
        ttk.Label(param_frame, text="Q值:").grid(row=0, column=4, padx=5)
        self.q_var = tk.DoubleVar(value=30.0)
        ttk.Entry(param_frame, textvariable=self.q_var, width=8, font=cn_font).grid(row=0, column=5, padx=5)
        ttk.Label(param_frame, text="LMS长度:").grid(row=0, column=6, padx=5)
        self.lms_len_var = tk.IntVar(value=32)
        ttk.Entry(param_frame, textvariable=self.lms_len_var, width=8, font=cn_font).grid(row=0, column=7, padx=5)
        ttk.Label(param_frame, text="LMS步长μ:").grid(row=0, column=8, padx=5)
        self.lms_mu_var = tk.DoubleVar(value=0.01)
        ttk.Entry(param_frame, textvariable=self.lms_mu_var, width=8, font=cn_font).grid(row=0, column=9, padx=5)
        # VAD开关
        self.vad_enable_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="启用VAD语音检测", variable=self.vad_enable_var).grid(row=0, column=10, padx=10)

        # 按钮区
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=5, padx=10)
        self.process_btn = ttk.Button(btn_frame, text="处理并分析", command=self.process_and_analyze)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        if HAS_REALTIME:
            self.realtime_btn = ttk.Button(btn_frame, text="实时处理", command=self.start_realtime)
            self.realtime_btn.pack(side=tk.LEFT, padx=5)

        # 结果展示区（上）
        result_frame = ttk.Frame(self)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.result_text = tk.Text(result_frame, height=12, wrap=tk.WORD, font=cn_font_text)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "等待文件拖入或选择...\n")

        # 大模型建议区（下，扩大高度）
        suggestion_frame = ttk.LabelFrame(self, text="大模型建议")
        suggestion_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.suggestion_text = tk.Text(suggestion_frame, height=10, wrap=tk.WORD, font=cn_font_text, bg='#f8fff8')
        self.suggestion_text.pack(fill=tk.BOTH, expand=True)
        self.suggestion_text.insert(tk.END, "大模型建议将在此显示...\n")

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)

    def drop(self, event):
        filepath = event.data.strip('{}')
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, filepath)

    def process_and_analyze(self):
        filepath = self.file_entry.get().strip()
        if not filepath or not os.path.isfile(filepath):
            messagebox.showerror("错误", "请选择有效的 WAV 文件")
            return
        if not filepath.lower().endswith(".wav"):
            messagebox.showerror("错误", "请拖入或选择 WAV 格式文件")
            return

        self.result_text.delete(1.0, tk.END)
        self.suggestion_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"处理文件: {filepath}\n")

        # 参数获取
        filter_type = self.filter_type_var.get()
        f0 = self.f0_var.get()
        Q = self.q_var.get()
        lms_len = self.lms_len_var.get()
        lms_mu = self.lms_mu_var.get()
        vad_enable = self.vad_enable_var.get()

        def safe_insert(widget, content, clear=False):
            def _insert():
                if clear:
                    widget.delete(1.0, tk.END)
                widget.insert(tk.END, content)
            widget.after(0, _insert)

        def worker():
            try:
                # 判断是否需要大模型判断
                if filter_type == "auto":
                    try:
                        features, result = classify_audio(filepath)
                    except Exception as e:
                        safe_insert(self.result_text, f"特征提取失败：{e}\n")
                        return

                    safe_insert(self.result_text, f"噪声分类结果: {result}\n")
                    safe_insert(self.result_text, f"STFT特征: {json.dumps(features, ensure_ascii=False, indent=2)}\n")

                    if result == "非稳态噪声":
                        filter_type_real = "lms"
                        method = "LMS 自适应滤波"
                    elif result == "稳态噪声":
                        filter_type_real = "iir"
                        method = "IIR 陷波滤波"
                    else:
                        filter_type_real = "iir"
                        method = "默认使用 IIR"
                else:
                    filter_type_real = filter_type
                    method = "手动选择: " + ("IIR 陷波滤波" if filter_type == "iir" else "LMS 自适应滤波")

                safe_insert(self.result_text, f"选择滤波器: {method}\n")

                out_path = process_audio(
                    filepath,
                    filter_type=filter_type_real,
                    f0=f0,
                    Q=Q,
                    filter_length=lms_len,
                    mu=lms_mu
                )
                if out_path:
                    safe_insert(self.result_text, f"滤波完成，输出文件: {out_path}\n")
                    plot_comparison(filepath, out_path, title_prefix="音频分析 - ", vad_enable=vad_enable)
                    y_orig, _ = sf.read(filepath)
                    y_filt, _ = sf.read(out_path)
                    min_len = min(len(y_orig), len(y_filt))
                    y_orig = y_orig[:min_len]
                    y_filt = y_filt[:min_len]
                    snr_before = 10 * np.log10(np.mean(y_orig ** 2) / (np.mean((y_orig - y_orig) ** 2) + 1e-10))
                    snr_after = 10 * np.log10(np.mean(y_orig ** 2) / (np.mean((y_orig - y_filt) ** 2) + 1e-10))
                    params = {
                        "filter_type": filter_type_real,
                        "f0": f0,
                        "Q": Q,
                        "lms_length": lms_len,
                        "lms_mu": lms_mu
                    }
                    try:
                        suggestion = get_filter_suggestion(snr_before, snr_after, filter_type_real, params)
                        safe_insert(self.suggestion_text, f"{suggestion}\n", clear=True)
                    except Exception as e:
                        safe_insert(self.suggestion_text, f"大模型建议获取失败：{e}\n", clear=True)
                else:
                    safe_insert(self.result_text, "滤波失败\n")
            except Exception as e:
                safe_insert(self.result_text, f"处理异常：{e}\n")

        threading.Thread(target=worker, daemon=True).start()

    def start_realtime(self):
        if not HAS_REALTIME:
            messagebox.showinfo("提示", "未检测到实时处理模块。")
            return
        filter_type = self.filter_type_var.get()
        f0 = self.f0_var.get()
        Q = self.q_var.get()
        lms_len = self.lms_len_var.get()
        lms_mu = self.lms_mu_var.get()
        vad_enable = self.vad_enable_var.get()

        rec_win = tk.Toplevel(self)
        rec_win.title("实时录音")
        rec_win.geometry("400x180")
        rec_win.resizable(False, False)
        rec_label = tk.Label(rec_win, text="点击开始录音，录音结束后自动处理", font=('微软雅黑', 12))
        rec_label.pack(pady=10)
        status_var = tk.StringVar(value="等待录音...")
        status_label = tk.Label(rec_win, textvariable=status_var, font=('微软雅黑', 11))
        status_label.pack(pady=5)
        record_btn = ttk.Button(rec_win, text="开始录音", width=15)
        record_btn.pack(pady=10)
        stop_btn = ttk.Button(rec_win, text="停止录音", width=15, state=tk.DISABLED)
        stop_btn.pack(pady=5)

        RATE = 16000
        CHANNELS = 1
        FORMAT = pyaudio.paInt16
        CHUNK = 1024

        self._recording = False
        self._audio_frames = []

        def safe_insert(widget, content, clear=False):
            def _insert():
                if clear:
                    widget.delete(1.0, tk.END)
                widget.insert(tk.END, content)
            widget.after(0, _insert)

        def record():
            self._audio_frames = []
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            status_var.set("录音中...点击停止录音")
            while self._recording:
                data = stream.read(CHUNK)
                self._audio_frames.append(data)
            stream.stop_stream()
            stream.close()
            p.terminate()
            status_var.set("录音结束，处理中...")

            # 保存到临时wav文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir='.') as tmpfile:
                wf = wave.open(tmpfile.name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self._audio_frames))
                wf.close()
                tmp_wav = tmpfile.name

            def process_recorded():
                try:
                    out_path = process_audio(
                        tmp_wav,
                        filter_type=filter_type if filter_type != "auto" else "iir",
                        f0=f0,
                        Q=Q,
                        filter_length=lms_len,
                        mu=lms_mu
                    )
                    if out_path:
                        safe_insert(self.result_text, f"实时录音已保存: {tmp_wav}\n", clear=True)
                        safe_insert(self.result_text, f"滤波完成，输出文件: {out_path}\n")
                        plot_comparison(tmp_wav, out_path, title_prefix="实时录音 - ", vad_enable=vad_enable)
                        y_orig, _ = sf.read(tmp_wav)
                        y_filt, _ = sf.read(out_path)
                        min_len = min(len(y_orig), len(y_filt))
                        y_orig = y_orig[:min_len]
                        y_filt = y_filt[:min_len]
                        snr_before = 10 * np.log10(np.mean(y_orig ** 2) / (np.mean((y_orig - y_orig) ** 2) + 1e-10))
                        snr_after = 10 * np.log10(np.mean(y_orig ** 2) / (np.mean((y_orig - y_filt) ** 2) + 1e-10))
                        params = {
                            "filter_type": filter_type if filter_type != "auto" else "iir",
                            "f0": f0,
                            "Q": Q,
                            "lms_length": lms_len,
                            "lms_mu": lms_mu
                        }
                        try:
                            suggestion = get_filter_suggestion(snr_before, snr_after, params["filter_type"], params)
                            safe_insert(self.suggestion_text, f"{suggestion}\n", clear=True)
                        except Exception as e:
                            safe_insert(self.suggestion_text, f"大模型建议获取失败：{e}\n", clear=True)
                    else:
                        safe_insert(self.result_text, "滤波失败\n")
                except Exception as e:
                    safe_insert(self.result_text, f"实时录音处理失败：{e}\n")
                rec_win.destroy()

            threading.Thread(target=process_recorded, daemon=True).start()

        def start_record():
            self._recording = True
            record_btn.config(state=tk.DISABLED)
            stop_btn.config(state=tk.NORMAL)
            threading.Thread(target=record, daemon=True).start()

        def stop_record():
            self._recording = False
            stop_btn.config(state=tk.DISABLED)
            status_var.set("正在保存录音...")

        record_btn.config(command=start_record)
        stop_btn.config(command=stop_record)

if __name__ == '__main__':
    app = AudioApp()
    app.mainloop()