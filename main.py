import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from extract_wav_features import classify_audio
import soundfile as sf
import numpy as np
from filter import process_audio
from plot_audio_comparison import plot_comparison
import threading
import tempfile
import wave
import pyaudio
import matplotlib.pyplot as plt

# 实时处理模块导入
try:
    from realtime_audio_processing import RealTimeAudioProcessor
    HAS_REALTIME = True
except ImportError:
    HAS_REALTIME = False

class AudioApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("音频噪声处理系统")
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

        # 创建输出目录
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # 噪声设置区
        noise_frame = ttk.LabelFrame(self, text="噪声生成设置")
        noise_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 噪声类型选择
        ttk.Label(noise_frame, text="噪声类型:").grid(row=0, column=0, padx=5, pady=2)
        self.noise_type_var = tk.StringVar(value="steady")
        ttk.Combobox(noise_frame, textvariable=self.noise_type_var, 
                    values=["steady", "non_steady"], width=12, 
                    state="readonly", font=cn_font).grid(row=0, column=1, padx=5)
        
        # 噪声参数
        ttk.Label(noise_frame, text="噪声频率:").grid(row=0, column=2, padx=5)
        self.noise_freq_var = tk.DoubleVar(value=50.0)
        ttk.Entry(noise_frame, textvariable=self.noise_freq_var, width=8, font=cn_font).grid(row=0, column=3, padx=5)
        
        ttk.Label(noise_frame, text="噪声强度:").grid(row=0, column=4, padx=5)
        self.noise_amp_var = tk.DoubleVar(value=0.5)
        ttk.Entry(noise_frame, textvariable=self.noise_amp_var, width=8, font=cn_font).grid(row=0, column=5, padx=5)

        # 按钮区
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=5, padx=10)
        self.add_noise_btn = ttk.Button(btn_frame, text="添加噪声", command=self.add_noise)
        self.add_noise_btn.pack(side=tk.LEFT, padx=5)
        self.process_btn = ttk.Button(btn_frame, text="自动滤波", command=self.process_and_analyze)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        if HAS_REALTIME:
            self.realtime_btn = ttk.Button(btn_frame, text="实时处理", command=self.start_realtime)
            self.realtime_btn.pack(side=tk.LEFT, padx=5)

        # 结果展示区
        result_frame = ttk.Frame(self)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.result_text = tk.Text(result_frame, height=15, wrap=tk.WORD, font=cn_font_text)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "等待文件拖入或选择...\n")

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)

    def drop(self, event):
        filepath = event.data.strip('{}')
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, filepath)

    def add_noise(self):
        filepath = self.file_entry.get().strip()
        if not filepath or not os.path.isfile(filepath):
            messagebox.showerror("错误", "请选择有效的 WAV 文件")
            return

        # 获取噪声参数
        noise_type = self.noise_type_var.get()
        noise_freq = self.noise_freq_var.get()
        noise_amp = self.noise_amp_var.get()

        try:
            # 拷贝原始（干净）音频到 output 目录，便于后续 SNR 分析
            filename = os.path.basename(filepath)
            clean_path = os.path.join(self.output_dir, filename)
            # 如果目标文件不存在则拷贝
            if not os.path.exists(clean_path):
                import shutil
                shutil.copy2(filepath, clean_path)
                self.result_text.insert(tk.END, f"已拷贝原始音频至: {clean_path}\n")
            
            # 读取原始音频
            y_orig, sr = sf.read(filepath)
            is_stereo = len(y_orig.shape) > 1 and y_orig.shape[1] > 1
            
            # 生成噪声
            t = np.arange(len(y_orig)) / sr
            if noise_type == "steady":
                noise = noise_amp * np.sin(2 * np.pi * noise_freq * t)
            else:
                noise = noise_amp * np.sin(2 * np.pi * noise_freq * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
            if is_stereo:
                noise = np.column_stack((noise, noise))
            
            # 混合噪声并保存带噪文件
            y_noisy = y_orig + noise
            noisy_path = os.path.join(self.output_dir, os.path.splitext(filename)[0] + '_noisy.wav')
            sf.write(noisy_path, y_noisy, sr)
            
            # 显示添加噪声消息
            self.result_text.insert(tk.END, f"已添加噪声，保存至: {noisy_path}\n")
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, noisy_path)
            
            # 同时计算并绘制滤波前 SNR（使用拷贝的干净音频作为参考）
            self.plot_snr(y_orig, y_noisy, sr)
            
        except Exception as e:
            messagebox.showerror("错误", f"添加噪声失败: {str(e)}")

    def plot_snr(self, clean, noisy, sr, window_size=1024):
        # 计算分段SNR
        snr_values = []
        for i in range(0, len(clean)-window_size, window_size//2):
            clean_segment = clean[i:i+window_size]
            noise_segment = noisy[i:i+window_size] - clean_segment
            if isinstance(clean_segment, np.ndarray) and len(clean_segment.shape) > 1:
                clean_segment = clean_segment.mean(axis=1)
            if isinstance(noise_segment, np.ndarray) and len(noise_segment.shape) > 1:
                noise_segment = noise_segment.mean(axis=1)
            signal_power = np.mean(clean_segment ** 2)
            noise_power = np.mean(noise_segment ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            snr_values.append(snr)

        # 绘制SNR曲线
        plt.figure(figsize=(10, 4))
        time_points = np.arange(len(snr_values)) * (window_size/2) / sr
        plt.plot(time_points, snr_values)
        plt.title('SNR随时间变化')
        plt.xlabel('时间 (秒)')
        plt.ylabel('SNR (dB)')
        plt.grid(True)
        plt.tight_layout()

        # 保存SNR图
        snr_plot_path = os.path.join(self.output_dir, 'snr_analysis.png')
        plt.savefig(snr_plot_path)
        plt.close()
        self.result_text.insert(tk.END, f"SNR分析图已保存至: {snr_plot_path}\n")

    def process_and_analyze(self):
        filepath = self.file_entry.get().strip()
        if not filepath or not filepath.lower().endswith("_noisy.wav"):
            messagebox.showerror("错误", "请先添加噪声")
            return

        def worker():
            try:
                # 从拷贝的干净文件获取原始信号（拷贝时存储在 output 目录）
                filename = os.path.basename(filepath)
                # 假设原始文件名就是去掉 _noisy.wav 后缀
                clean_path = os.path.join(self.output_dir, filename.replace('_noisy.wav', '.wav'))
                if not os.path.exists(clean_path):
                    messagebox.showerror("错误", f"未找到原始干净音频文件：{clean_path}")
                    return
                y_clean, sr = sf.read(clean_path)
                y_noisy, _ = sf.read(filepath)

                # 获取滤波器参数和分类结果
                features, result = classify_audio(filepath)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"噪声分类结果: {result}\n")
                filter_type = "iir" if result == "稳态噪声" else "lms"
                
                filename = os.path.basename(filepath)
                out_path = os.path.join(self.output_dir, os.path.splitext(filename)[0] + '_filtered.wav')
                
                processed = process_audio(
                    input_path=filepath,
                    output_path=out_path,
                    filter_type=filter_type,
                    f0=self.noise_freq_var.get(),
                    Q=5.0 if filter_type == "iir" else None,
                    filter_length=64 if filter_type == "lms" else None,
                    mu=0.001 if filter_type == "lms" else None
                )

                if processed:
                    self.result_text.insert(tk.END, f"滤波完成，输出文件: {out_path}\n")
                    plot_comparison(filepath, out_path, title_prefix="频谱对比 - ")
                    
                    # 计算滤波前后 SNR（均以干净音频作为参考）
                    snr_before = np.mean([10 * np.log10(np.mean(y_clean**2) / (np.mean((y_noisy - y_clean)**2)+1e-10))])
                    y_filt, _ = sf.read(out_path)
                    snr_after = np.mean([10 * np.log10(np.mean(y_clean**2) / (np.mean((y_filt - y_clean)**2)+1e-10))])
                    self.result_text.insert(tk.END, f"滤波前 SNR: {snr_before:.2f} dB\n滤波后 SNR: {snr_after:.2f} dB\n")
                else:
                    self.result_text.insert(tk.END, "滤波失败\n")
            except Exception as e:
                self.result_text.insert(tk.END, f"处理失败: {str(e)}\n")

        threading.Thread(target=worker, daemon=True).start()

    def start_realtime(self):
        if not HAS_REALTIME:
            messagebox.showinfo("提示", "未检测到实时处理模块。")
            return

        noise_type = self.noise_type_var.get()
        noise_freq = self.noise_freq_var.get()
        noise_amp = self.noise_amp_var.get()

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

        def safe_insert(widget, content, clear=False):
            def _insert():
                if clear:
                    widget.delete(1.0, tk.END)
                widget.insert(tk.END, content)
            widget.after(0, _insert)

        def process_recorded_audio(tmp_wav):
            try:
                y_orig, sr = sf.read(tmp_wav)
                t = np.arange(len(y_orig)) / sr
                if noise_type == "steady":
                    noise = noise_amp * np.sin(2 * np.pi * noise_freq * t)
                else:
                    noise = noise_amp * np.sin(2 * np.pi * noise_freq * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
                y_noisy = y_orig + noise
                noisy_path = os.path.join(self.output_dir, 'realtime_noisy.wav')
                sf.write(noisy_path, y_noisy, sr)
                
                from plot_audio_comparison import plot_comparison
                features, result = classify_audio(noisy_path)
                filter_type = "iir" if result == "稳态噪声" else "lms"
                params = {
                    "filter_type": filter_type,
                    "f0": noise_freq,
                    "Q": 5.0 if filter_type == "iir" else None,
                    "filter_length": 64 if filter_type == "lms" else None,
                    "mu": 0.001 if filter_type == "lms" else None
                }
                out_path = os.path.join(self.output_dir, 'realtime_filtered.wav')
                if process_audio(noisy_path, out_path, **params):
                    safe_insert(self.result_text, f"实时处理完成\n原始录音: {tmp_wav}\n带噪声音频: {noisy_path}\n滤波后音频: {out_path}\n", clear=True)
                    plot_comparison(noisy_path, out_path, title_prefix="实时处理 - ")
                    self.plot_snr(y_orig, y_noisy, sr)
                else:
                    safe_insert(self.result_text, "滤波失败\n")
            except Exception as e:
                safe_insert(self.result_text, f"实时处理失败：{str(e)}\n")
            finally:
                rec_win.destroy()

        def record():
            try:
                self._recording = True
                self._audio_frames = []
                p = pyaudio.PyAudio()
                stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                input=True, frames_per_buffer=CHUNK)

                def record_thread():
                    while self._recording:
                        try:
                            if stream.is_active():
                                data = stream.read(CHUNK, exception_on_overflow=False)
                                self._audio_frames.append(data)
                                time.sleep(0.01)  # yield control
                        except Exception as e:
                            print(f"录音错误: {e}")
                            break
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    if len(self._audio_frames) > 0:
                        self.after(100, lambda: process_recorded_audio(self._save_temp()))

                threading.Thread(target=record_thread, daemon=True).start()
            except Exception as e:
                messagebox.showerror("错误", f"录音初始化失败: {str(e)}")

        def stop_record():
            self._recording = False
            status_var.set("正在保存录音...")
            stop_btn.config(state=tk.DISABLED)

        def start_record():
            record_btn.config(state=tk.DISABLED)
            stop_btn.config(state=tk.NORMAL)
            stop_btn.config(command=stop_record)
            record()

        self._save_temp = _save_temp.__get__(self)

        # Helper function to save recorded frames to a temporary file in output
        def _save_temp(self):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=self.output_dir) as tmpfile:
                wf = wave.open(tmpfile.name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self._audio_frames))
                wf.close()
                return tmpfile.name

        record_btn.config(command=start_record)

if __name__ == '__main__':
    app = AudioApp()
    app.mainloop()