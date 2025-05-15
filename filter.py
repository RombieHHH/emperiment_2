"""
Audio Filter Module

提供 IIR 陷波滤波器和 LMS 自适应滤波器，对单声道或立体声音频进行降噪处理。
留出关键参数接口，方便在其他脚本中调用。
"""
import os
import sys
from pathlib import Path
import numpy as np
from scipy import signal
import soundfile as sf
import librosa


class AudioFilter:
    """
    音频滤波器类，支持 IIR 陷波滤波和 LMS 自适应滤波。

    Attributes:
        filter_length (int): LMS 滤波器长度（tap 数）。
        mu (float): LMS 步长因子。
    """

    def __init__(self, filter_length: int = 64, mu: float = 0.001):
        """
        初始化滤波器参数。

        Args:
            filter_length: LMS 滤波器 tap 长度，默认为 64。
            mu: LMS 学习率（步长），默认为 0.001。
        """
        self.filter_length = filter_length
        self.mu = mu
        self.w = None  # 保存滤波器权重

    def _detect_bursts(self, data: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        """改进的突发检测，使用短时能量和过零率双重检测"""
        frame_length = 1024
        hop_length = frame_length // 2
        
        # 计算短时能量
        energy = np.array([
            np.sum(data[i:i+frame_length]**2) 
            for i in range(0, len(data)-frame_length, hop_length)
        ])
        energy = energy / np.max(energy)
        
        # 计算过零率
        zero_crossings = np.array([
            np.sum(np.abs(np.diff(np.signbit(data[i:i+frame_length])))) 
            for i in range(0, len(data)-frame_length, hop_length)
        ])
        zero_crossings = zero_crossings / np.max(zero_crossings)
        
        # 综合判断突发段
        burst_frames = (energy > threshold) | (zero_crossings > threshold * 0.8)
        return burst_frames, range(0, len(data)-frame_length, hop_length)

    def lms_filter(self, data: np.ndarray, reference: np.ndarray, mu: float = None) -> np.ndarray:
        """改进的LMS滤波器，针对非稳态噪声优化"""
        if mu is not None:
            self.mu = mu

        # 确保输入为一维数组
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # 初始化
        if self.w is None:
            self.w = np.zeros(self.filter_length)
        y = np.zeros_like(data)
        
        # 检测突发段
        burst_frames, frame_indices = self._detect_bursts(data)
        
        # 计算自适应步长
        adaptive_mu = np.ones(len(data)) * self.mu
        for i, frame_start in enumerate(frame_indices):
            if burst_frames[i]:
                frame_end = min(frame_start + 1024, len(data))
                # 突发段使用更大的步长和更短的滤波器长度
                adaptive_mu[frame_start:frame_end] = self.mu * 3
                
        # 改进的LMS算法
        for n in range(self.filter_length, len(data)):
            x = reference[n-self.filter_length:n][::-1]
            power = np.dot(x, x) + 1e-10
            
            # 计算输出
            y[n] = np.dot(self.w, x)
            
            # 计算误差
            e = data[n] - y[n]
            
            # 自适应更新权重
            self.w += 2 * adaptive_mu[n] * e * x / power
            
            # 在突发段后重置权重
            if n > 0 and adaptive_mu[n] > adaptive_mu[n-1]:
                self.w = np.zeros(self.filter_length)

        return data - y

    def _estimate_noise_frequency(self, data: np.ndarray, fs: int) -> float:
        """估计噪声主频率"""
        D = librosa.stft(data)
        freqs = librosa.fft_frequencies(sr=fs)
        magnitude_spectrum = np.abs(D)
        avg_magnitude = np.mean(magnitude_spectrum, axis=1)
        peak_freq_idx = np.argmax(avg_magnitude)
        return freqs[peak_freq_idx]

    def generate_reference(self, length: int, fs: int, f0: float = 50.0) -> np.ndarray:
        """
        生成正弦作为参考噪声。

        Args:
            length: 信号长度（样本数）。
            fs: 采样率（Hz）。
            f0: 噪声频率，默认为工频 50Hz。

        Returns:
            一维参考信号。
        """
        t = np.arange(length) / fs
        return np.sin(2 * np.pi * f0 * t)

    def iir_notch_filter(self, data: np.ndarray, fs: int, f0: float = 50.0, Q: float = 5.0) -> np.ndarray:
        """
        改进的IIR陷波滤波器，增加自适应Q值

        Args:
            data: 输入信号（1D 或 2D 立体声）。
            fs: 采样率（Hz）。
            f0: 陷波频率，默认为 50Hz。
            Q: 品质因数，值越高带阻越窄，默认为 5。

        Returns:
            滤波后信号。与输入形状相同。
        """
        # 检测突发段
        burst_frames = self._detect_bursts(data)

        # 在突发段降低Q值以增加带宽
        filtered = np.zeros_like(data)
        for i in range(len(burst_frames)):
            start = i * 512
            end = start + 1024
            if end > len(data):
                end = len(data)

            local_Q = Q if not burst_frames[i] else Q/2
            w0 = f0 / (fs/2)
            b, a = signal.iirnotch(w0, local_Q)

            if start == 0:
                filtered[start:end] = signal.filtfilt(b, a, data[start:end])
            else:
                # 处理段间过渡
                overlap = 100
                temp = signal.filtfilt(b, a, data[start-overlap:end])
                filtered[start:end] = temp[overlap:]

        return filtered


# 全局函数：批量处理并保存结果

def process_audio(input_path, output_path, filter_type, f0, Q=None, filter_length=None, mu=None):
    """
    读取音频文件，应用指定滤波器并保存结果。

    Args:
        input_path: 输入 WAV 文件路径。
        output_path: 输出 WAV 文件路径。
        filter_type: 'iir' 或 'lms'。
        f0: 陷波中心频率，用于 IIR。
        Q: IIR 品质因数。
        filter_length: LMS 滤波器长度。
        mu: LMS 学习率。

    Returns:
        输出文件路径，若失败返回 None。
    """
    try:
        data, fs = sf.read(input_path)
        # 对于非稳态噪声，使用更激进的参数
        if filter_type.lower() == 'lms':
            filter_length = 32  # 减小滤波器长度以提高响应速度
            mu = 0.01  # 增大步长以加快收敛
        else:
            Q = 2.0  # 降低Q值以增加带宽
            
        audio_filter = AudioFilter(filter_length=filter_length, mu=mu)
        
        # 单声道转换
        if len(data.shape) > 1:
            data = data.mean(axis=1)
            
        if filter_type.lower() == 'lms':
            # 自动估计主频率
            estimated_f0 = audio_filter._estimate_noise_frequency(data, fs)
            f0 = estimated_f0 if f0 is None else f0
            
            # 生成复合参考信号
            t = np.arange(len(data)) / fs
            reference = np.sin(2 * np.pi * f0 * t)
            # 添加谐波分量
            reference += 0.5 * np.sin(4 * np.pi * f0 * t)
            
            filtered = audio_filter.lms_filter(data, reference, mu)
        else:
            filtered = audio_filter.iir_notch_filter(data, fs, f0, Q)
            
        sf.write(output_path, filtered, fs)
        return output_path
        
    except Exception as e:
        print(f"处理音频文件时出错: {e}")
        return None


# 示例调用
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="音频降噪：IIR 陷波 / LMS 自适应滤波")
    parser.add_argument('input_file', help='输入 WAV 文件路径')
    parser.add_argument('output_file', help='输出 WAV 文件路径')
    parser.add_argument('--filter', choices=['iir', 'lms'], default='iir',
                        help='滤波类型，默认为 iir')
    parser.add_argument('--f0', type=float, default=50.0,
                        help='陷波中心频率（Hz），用于 IIR 和 LMS 参考信号')
    parser.add_argument('--Q', type=float, default=30.0,
                        help='IIR 品质因数')
    parser.add_argument('--length', type=int, default=32,
                        help='LMS 滤波器长度')
    parser.add_argument('--mu', type=float, default=0.01,
                        help='LMS 步长因子')
    args = parser.parse_args()

    result = process_audio(
        input_path=args.input_file,
        output_path=args.output_file,
        filter_type=args.filter,
        f0=args.f0,
        Q=args.Q,
        filter_length=args.length,
        mu=args.mu
    )
    print(f"输出文件: {result}")
