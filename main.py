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
        # 初始化父类
        TkinterDnD.Tk.__init__(self)
        self.init_window()
        self.init_variables()
        self.create_widgets()
        self.setup_output_dir()

    def init_window(self):
        """初始化窗口基本设置"""
        self.title("音频噪声处理系统")
        self.geometry("1024x768")
        self.configure(bg='#f4f4f4')
        
        # 创建主容器
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 设置样式
        style = ttk.Style()
        style.theme_use('clam')
        self.setup_styles(style)

    def init_variables(self):
        """初始化变量"""
        self.noise_type_var = tk.StringVar(value="steady")
        self.noise_freq_var = tk.DoubleVar(value=50.0)
        self.noise_amp_var = tk.DoubleVar(value=0.5)
        self.sample_rate_var = tk.StringVar(value="44100")
        self.q_factor_var = tk.DoubleVar(value=30.0)
        self.filter_length_var = tk.IntVar(value=64)

    def setup_styles(self, style):
        """设置界面样式"""
        self.cn_font = ('微软雅黑', 12)
        self.cn_font_bold = ('微软雅黑', 13, 'bold')
        self.cn_font_text = ('微软雅黑', 11)
        
        style.configure('Custom.TFrame', background='#f4f4f4')
        style.configure('Custom.TButton', 
                       font=self.cn_font, 
                       padding=6)
        style.configure('Custom.TLabel', 
                       font=self.cn_font, 
                       background='#f4f4f4')
        style.configure('Custom.TLabelframe', 
                       background='#f4f4f4',
                       padding=10)
        style.configure('Custom.TLabelframe.Label', 
                       font=self.cn_font_bold)

    def setup_output_dir(self):
        """设置输出目录"""
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def create_widgets(self):
        """创建界面组件"""
        # 文件选择区
        self.create_file_section()
        # 拖拽区域
        self.create_drop_zone()
        # 参数设置区
        self.create_parameter_section()
        # 按钮区
        self.create_button_section()
        # 结果显示区
        self.create_result_section()

    def create_file_section(self):
        """创建文件选择区域"""
        file_frame = ttk.Frame(self.main_container, style='Custom.TFrame')
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="音频文件:", 
                 style='Custom.TLabel').pack(side=tk.LEFT)
        self.file_entry = ttk.Entry(file_frame, width=60, 
                                  font=self.cn_font)
        self.file_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="选择文件", 
                  command=self.browse_file,
                  style='Custom.TButton').pack(side=tk.LEFT, padx=5)

    def create_drop_zone(self):
        """创建拖拽区域"""
        self.drop_label = tk.Label(self.main_container, text="或将音频文件拖拽到此处", bg='#e0e0e0', fg='black',
                                   font=self.cn_font_bold, relief="groove", bd=2, width=60, height=2)
        self.drop_label.pack(pady=5)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.drop)

    def create_parameter_section(self):
        """创建参数设置区"""
        # 噪声设置区
        noise_frame = ttk.LabelFrame(self.main_container, text="噪声生成设置", style='Custom.TLabelframe')
        noise_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 噪声类型选择
        ttk.Label(noise_frame, text="噪声类型:", style='Custom.TLabel').grid(row=0, column=0, padx=5, pady=2)
        ttk.Combobox(noise_frame, textvariable=self.noise_type_var, 
                    values=["steady", "non_steady"], width=12, 
                    state="readonly", font=self.cn_font).grid(row=0, column=1, padx=5)
        
        # 噪声参数
        ttk.Label(noise_frame, text="噪声频率:", style='Custom.TLabel').grid(row=0, column=2, padx=5)
        ttk.Entry(noise_frame, textvariable=self.noise_freq_var, width=8, font=self.cn_font).grid(row=0, column=3, padx=5)
        
        ttk.Label(noise_frame, text="噪声强度:", style='Custom.TLabel').grid(row=0, column=4, padx=5)
        ttk.Entry(noise_frame, textvariable=self.noise_amp_var, width=8, font=self.cn_font).grid(row=0, column=5, padx=5)

        # 高级参数区域
        advanced_frame = ttk.LabelFrame(self.main_container, text="高级参数设置", style='Custom.TLabelframe')
        advanced_frame.pack(fill=tk.X, padx=15, pady=(10,5))
        
        # 网格布局参数设置
        param_frame = ttk.Frame(advanced_frame, style='Custom.TFrame')
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 第一行参数
        ttk.Label(param_frame, text="采样率(Hz):", style='Custom.TLabel').grid(row=0, column=0, padx=5, pady=2)
        ttk.Entry(param_frame, textvariable=self.sample_rate_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(param_frame, text="Q值:", style='Custom.TLabel').grid(row=0, column=2, padx=5)
        ttk.Entry(param_frame, textvariable=self.q_factor_var, width=8).grid(row=0, column=3, padx=5)
        
        ttk.Label(param_frame, text="滤波器长度:", style='Custom.TLabel').grid(row=0, column=4, padx=5)
        ttk.Entry(param_frame, textvariable=self.filter_length_var, width=8).grid(row=0, column=5, padx=5)

    def create_button_section(self):
        """创建按钮区"""
        btn_frame = ttk.Frame(self.main_container, style='Custom.TFrame')
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.add_noise_btn = ttk.Button(btn_frame, text="添加噪声", command=self.add_noise, style='Custom.TButton')
        self.add_noise_btn.pack(side=tk.LEFT, padx=5)
        self.process_btn = ttk.Button(btn_frame, text="自动滤波", command=self.process_and_analyze, style='Custom.TButton')
        self.process_btn.pack(side=tk.LEFT, padx=5)
        if HAS_REALTIME:
            self.realtime_btn = ttk.Button(btn_frame, text="实时处理", command=self.start_realtime, style='Custom.TButton')
            self.realtime_btn.pack(side=tk.LEFT, padx=5)

    def create_result_section(self):
        """创建结果显示区"""
        result_frame = ttk.Frame(self.main_container, style='Custom.TFrame')
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.result_text = tk.Text(result_frame, height=15, wrap=tk.WORD, font=self.cn_font_text)
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

        try:
            # 读取原始音频
            y_orig, sr = sf.read(filepath)
            filename = os.path.basename(filepath)
            base_name = os.path.splitext(filename)[0]
            clean_path = os.path.join(self.output_dir, filename)
            
            # 保存原始音频副本
            if not os.path.exists(clean_path):
                import shutil
                shutil.copy2(filepath, clean_path)
                self.result_text.insert(tk.END, f"已拷贝原始音频至: {clean_path}\n")
            
            # 生成纯噪声信号
            noise_params = {
                'type': self.noise_type_var.get(),
                'frequency': self.noise_freq_var.get(),
                'amplitude': self.noise_amp_var.get(),
                'sample_rate': sr,
                'duration': len(y_orig) / sr
            }
            
            # 生成并保存噪声文件
            from noise_generator import generate_noise
            noise = generate_noise(**noise_params)
            noise_path = os.path.join(self.output_dir, f'{base_name}_noise.wav')
            sf.write(noise_path, noise, sr)
            
            # 分析噪声特征并保存
            features, noise_type = classify_audio(noise_path)
            params_path = os.path.join(self.output_dir, f'{base_name}_noise_params.json')
            
            # 保存噪声参数和分类结果
            with open(params_path, 'w') as f:
                json.dump({
                    'params': noise_params,
                    'features': features,
                    'classified_type': noise_type
                }, f, indent=2)
                
            # 合成带噪音频
            if len(y_orig.shape) > 1 and y_orig.shape[1] > 1:
                noise = np.column_stack((noise, noise))
            y_noisy = y_orig + noise
            noisy_path = os.path.join(self.output_dir, f'{base_name}_noisy.wav')
            sf.write(noisy_path, y_noisy, sr)
            
            self.result_text.insert(tk.END, 
                f"已生成噪声文件: {noise_path}\n"
                f"噪声类型识别结果: {noise_type}\n"
                f"带噪声音频已保存至: {noisy_path}\n"
                f"噪声参数和特征已保存至: {params_path}\n")
            
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, noisy_path)
            
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
                # 获取相关文件路径
                filename = os.path.basename(filepath)
                base_name = os.path.splitext(filename)[0].replace('_noisy', '')
                clean_path = os.path.join(self.output_dir, f'{base_name}.wav')
                noise_path = os.path.join(self.output_dir, f'{base_name}_noise.wav')
                params_path = os.path.join(self.output_dir, f'{base_name}_noise_params.json')
                
                # 检查必要文件是否存在
                for path in [clean_path, noise_path, params_path]:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"找不到文件: {path}")
                
                # 读取噪声参数和分类结果
                with open(params_path, 'r') as f:
                    noise_info = json.load(f)
                
                # 处理音频
                filter_params = {
                    'input_path': filepath,
                    'output_path': os.path.join(self.output_dir, f'{base_name}_filtered.wav'),
                    'noise_path': noise_path,
                    'filter_type': 'iir' if noise_info['classified_type'] == "稳态噪声" else 'lms',
                    'noise_params': noise_info['params']
                }
                
                processed = process_audio(**filter_params)

                if processed:
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, f"滤波完成，输出文件: {filter_params['output_path']}\n")
                    
                    # 分析处理效果
                    snr_results = plot_comparison(
                        original_path=filepath,
                        processed_path=filter_params['output_path'],
                        clean_path=clean_path,
                        noise_path=noise_path,
                        title_prefix=f"频谱对比 ({noise_info['classified_type']}) - "
                    )
                    
                    if snr_results:
                        snr_before, snr_after = snr_results
                        self.result_text.insert(tk.END, 
                            f"平均SNR分析:\n滤波前: {snr_before:.2f} dB\n滤波后: {snr_after:.2f} dB\n"
                            f"SNR提升: {snr_after - snr_before:.2f} dB\n")
                else:
                    self.result_text.insert(tk.END, "滤波失败\n")
                    
            except Exception as e:
                self.result_text.insert(tk.END, f"处理失败: {str(e)}\n")

        threading.Thread(target=worker, daemon=True).start()

    def start_realtime(self):
        if not HAS_REALTIME:
            messagebox.showinfo("提示", "未检测到实时处理模块。")
            return
            
        # 定义录音参数
        FORMAT = pyaudio.paInt16  # 改用PCM格式
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024

        rec_win = tk.Toplevel(self)
        rec_win.title("实时录音")
        rec_win.geometry("400x180")
        rec_win.resizable(False, False)
        
        # 录音界面控件
        rec_label = tk.Label(rec_win, text="点击开始录音，录音结束后自动处理", font=('微软雅黑', 12))
        rec_label.pack(pady=10)
        status_var = tk.StringVar(value="等待录音...")
        status_label = tk.Label(rec_win, textvariable=status_var, font=('微软雅黑', 11))
        status_label.pack(pady=5)
        record_btn = ttk.Button(rec_win, text="开始录音", width=15)
        record_btn.pack(pady=10)
        stop_btn = ttk.Button(rec_win, text="停止录音", width=15, state=tk.DISABLED)
        stop_btn.pack(pady=5)

        self._recording = False  # 录音状态标志
        self._audio_frames = []  # 音频数据缓存

        def record():
            try:
                p = pyaudio.PyAudio()
                stream = p.open(format=FORMAT, 
                              channels=CHANNELS,
                              rate=RATE,
                              input=True, 
                              frames_per_buffer=CHUNK)
                
                self._recording = True
                self._audio_frames = []
                
                def record_thread():
                    try:
                        while self._recording:
                            data = stream.read(CHUNK, exception_on_overflow=False)
                            self._audio_frames.append(data)
                            import time
                            time.sleep(0.001)  # 短暂释放CPU
                    finally:
                        stream.stop_stream()
                        stream.close()
                        p.terminate()
                        
                        if len(self._audio_frames) > 0:
                            # 在主线程中处理录音数据
                            rec_win.after(100, process_recorded_audio)
                
                # 启动录音线程
                threading.Thread(target=record_thread, daemon=True).start()
                
                # 更新界面状态
                status_var.set("录音中...点击停止录音")
                record_btn.config(state=tk.DISABLED)
                stop_btn.config(state=tk.NORMAL)  # 修复这里的语法错误
                
            except Exception as e:
                messagebox.showerror("错误", f"录音初始化失败: {str(e)}")
                self._recording = False

        def stop_recording():
            self._recording = False
            status_var.set("正在保存录音...")
            stop_btn.config(state=tk.DISABLED)

        def process_recorded_audio():
            try:
                from io import BytesIO
                import wave
                
                # 保存为WAV格式
                wav_buffer = BytesIO()
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(self._audio_frames))
                
                # 转换为float32进行处理
                wav_buffer.seek(0)
                y_orig, sr = sf.read(wav_buffer)
                
                # 保存原始录音
                clean_path = os.path.join(self.output_dir, 'realtime_clean.wav')
                sf.write(clean_path, y_orig, sr)
                
                # 生成并添加噪声
                from noise_generator import add_noise_to_audio
                y_noisy = add_noise_to_audio(
                    y_orig, 
                    self.noise_type_var.get(),
                    self.noise_freq_var.get(),
                    self.noise_amp_var.get(),
                    sr
                )
                
                noisy_path = os.path.join(self.output_dir, 'realtime_noisy.wav')
                sf.write(noisy_path, y_noisy, sr)
                
                # 其余处理逻辑保持不变
                features, result = classify_audio(noisy_path)
                filter_type = "iir" if result == "稳态噪声" else "lms"
                out_path = os.path.join(self.output_dir, 'realtime_filtered.wav')
                
                if process_audio(noisy_path, out_path, 
                                   filter_type=filter_type,
                                   f0=self.noise_freq_var.get(),
                                   Q=5.0 if filter_type == "iir" else None,
                                   filter_length=64 if filter_type == "lms" else None,
                                   mu=0.001 if filter_type == "lms" else None):
                    # 更新处理结果
                    self.result_text.delete(1.0, tk.END)
                    self.result_text.insert(tk.END, 
                        f"实时处理完成\n"
                        f"原始录音: {clean_path}\n"
                        f"带噪声音频: {noisy_path}\n"
                        f"滤波后音频: {out_path}\n")
                    # 使用干净录音进行对比分析
                    plot_comparison(noisy_path, out_path, clean_path, title_prefix="实时处理 - ")
                        
            except Exception as e:
                messagebox.showerror("错误", f"处理录音失败: {str(e)}")
            finally:
                rec_win.destroy()

        # 绑定按钮事件
        record_btn.config(command=record)
        stop_btn.config(command=stop_recording)
        
        # 确保窗口关闭时停止录音
        rec_win.protocol("WM_DELETE_WINDOW", lambda: [stop_recording(), rec_win.destroy()])

if __name__ == '__main__':
    app = AudioApp()
    app.mainloop()