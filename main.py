import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from extract_wav_features import classify_audio
from filter import process_audio
from plot_audio_comparison import plot_comparison


class AudioApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("音频噪声分类与滤波")
        self.geometry("600x300")
        self.configure(bg='#f4f4f4')

        self.label = tk.Label(self, text="将音频文件拖拽到此处", bg='#f4f4f4', fg='black', font=('Arial', 14), relief="solid", bd=2, width=50, height=6)
        self.label.pack(pady=20)
        self.label.drop_target_register(DND_FILES)
        self.label.dnd_bind('<<Drop>>', self.drop)

        self.result_text = tk.Text(self, height=8, wrap=tk.WORD)
        self.result_text.pack(padx=10, fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "等待文件拖入...\n")

    def drop(self, event):
        filepath = event.data.strip('{}')
        if not filepath.lower().endswith(".wav"):
            messagebox.showerror("错误", "请拖入 WAV 格式文件")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"处理文件: {filepath}\n")

        try:
            features, result = classify_audio(filepath)
        except Exception as e:
            messagebox.showerror("错误", f"特征提取失败: {e}")
            return

        self.result_text.insert(tk.END, f"噪声分类结果: {result}\n")
        self.result_text.insert(tk.END, f"STFT特征: {json.dumps(features, ensure_ascii=False, indent=2)}\n")

        if result == "非稳态噪声":
            filter_type = "lms"
            method = "LMS 自适应滤波"
        elif result == "稳态噪声":
            filter_type = "iir"
            method = "IIR 陷波滤波"
        else:
            filter_type = "iir"
            method = "默认使用 IIR"

        self.result_text.insert(tk.END, f"选择滤波器: {method}\n")

        try:
            out_path = process_audio(filepath, filter_type=filter_type)
            if out_path:
                self.result_text.insert(tk.END, f"滤波完成，输出文件: {out_path}\n")
                # 绘图分析
                plot_comparison(filepath, out_path, title_prefix="音频分析 - ")
            else:
                self.result_text.insert(tk.END, "滤波失败\n")
        except Exception as e:
            messagebox.showerror("错误", f"滤波处理失败: {e}")


if __name__ == '__main__':
    app = AudioApp()
    app.mainloop()