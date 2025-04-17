import numpy as np
from scipy import signal
import soundfile as sf
import os
from pathlib import Path

class AudioFilter:
    def __init__(self):
        self.filter_length = 32
        self.mu = 0.01

    def lms_filter(self, data, reference, mu=None):
        """
        LMS自适应滤波器
        """
        if mu is not None:
            self.mu = mu

        # 检查是否为立体声
        is_stereo = len(data.shape) > 1
        if is_stereo:
            # 分别处理左右声道
            left_channel = self.lms_process_channel(data[:, 0], reference)
            right_channel = self.lms_process_channel(data[:, 1], reference)
            return np.column_stack((left_channel, right_channel))
        else:
            return self.lms_process_channel(data, reference)

    def lms_process_channel(self, channel_data, reference):
        """
        处理单个声道的LMS滤波
        """
        w = np.zeros(self.filter_length)
        y = np.zeros(len(channel_data))

        for n in range(self.filter_length, len(channel_data)):
            x = reference[n-self.filter_length:n][::-1]
            y[n] = np.dot(w, x)
            e = channel_data[n] - y[n]
            w = w + 2 * self.mu * e * x

        return channel_data - y

    def generate_reference(self, length, fs, f0=50):
        """
        生成参考噪声信号
        """
        t = np.arange(length) / fs
        reference = np.sin(2 * np.pi * f0 * t)
        # 确保reference是一维数组
        return reference.flatten()

    def iir_notch_filter(self, data, fs, f0=50, Q=30):
        """
        IIR陷波器
        params:
            data: 输入信号
            fs: 采样率
            f0: 中心频率
            Q: 品质因数
        """
        # 确保数据是一维数组
        is_stereo = len(data.shape) > 1
        if is_stereo:
            left_channel = self.iir_process_channel(data[:, 0], fs, f0, Q)
            right_channel = self.iir_process_channel(data[:, 1], fs, f0, Q)
            return np.column_stack((left_channel, right_channel))
        else:
            return self.iir_process_channel(data, fs, f0, Q)

    def iir_process_channel(self, channel_data, fs, f0, Q):
        """
        处理单个声道的IIR滤波
        """
        w0 = f0 / (fs / 2)
        b, a = signal.iirnotch(w0, Q)

        try:
            # 先尝试使用filtfilt
            filtered_data = signal.filtfilt(b, a, channel_data)
        except ValueError:
            try:
                # 如果filtfilt失败，尝试使用lfilter
                filtered_data = signal.lfilter(b, a, channel_data)
            except Exception as e:
                print(f"滤波失败: {str(e)}，返回原始信号")
                filtered_data = channel_data

        return filtered_data

def process_audio(input_path, filter_type='iir'):
    try:
        # 创建输出目录
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)

        # 读取音频文件
        data, fs = sf.read(input_path)

        # 检查音频数据
        if len(data) == 0:
            raise ValueError("音频文件为空")

        # 初始化滤波器
        audio_filter = AudioFilter()

        # 应用滤波器
        if filter_type.lower() == 'iir':
            filtered_data = audio_filter.iir_notch_filter(data, fs)
            suffix = 'IIR'
        else:
            reference = audio_filter.generate_reference(len(data), fs)
            filtered_data = audio_filter.lms_filter(data, reference)
            suffix = 'LMS'

        # 构建输出文件路径
        input_filename = Path(input_path).stem
        output_path = output_dir / f"{input_filename}({suffix}).wav"

        # 保存处理后的音频
        sf.write(output_path, filtered_data, fs)
        return str(output_path)

    except Exception as e:
        print(f"处理音频文件时出错: {str(e)}")
        return None

# 使用示例
if __name__ == "__main__":
    input_file = "C:/Users/Rombie/Downloads/archive/fold8/91209-5-1-2.wav"

    # IIR滤波
    iir_output = process_audio(input_file, 'iir')
    print(f"IIR滤波输出: {iir_output}")

    # LMS滤波
    lms_output = process_audio(input_file, 'lms')
    print(f"LMS滤波输出: {lms_output}")