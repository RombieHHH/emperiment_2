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

def process_audio(input_path, output_path, noise_path, filter_type, noise_params, **kwargs):
    """
    根据噪声参数和噪声文件进行针对性滤波
    
    Args:
        input_path: 输入的带噪音频路径
        output_path: 输出的滤波后音频路径
        noise_path: 纯噪声文件路径
        filter_type: 滤波器类型 ('iir' 或 'lms')
        noise_params: 噪声参数字典
    """
    # 读取音频文件
    audio, sr = sf.read(input_path)
    noise, _ = sf.read(noise_path)
    
    # 确保音频长度一致
    min_length = min(len(audio), len(noise))
    audio = audio[:min_length]
    noise = noise[:min_length]
    
    if filter_type == 'iir':
        # 使用纯噪声信号优化IIR滤波器参数
        f0 = noise_params['frequency']
        Q = kwargs.get('Q', 30.0)
        
        # 通过分析纯噪声信号调整Q值
        noise_spectrum = np.abs(np.fft.fft(noise))
        peak_idx = np.argmax(noise_spectrum[:len(noise_spectrum)//2])
        peak_width = np.sum(noise_spectrum > np.max(noise_spectrum) * 0.707)
        Q = min(Q, sr / (2 * peak_width)) # 根据噪声频谱宽度调整Q值
        
        w0 = f0 / (sr/2)
        b, a = signal.iirnotch(w0, Q)
        filtered = signal.filtfilt(b, a, audio)
        
    else:  # lms
        # 使用实际噪声作为参考
        filter_length = kwargs.get('filter_length', 32)
        mu = kwargs.get('mu', 0.02)
        
        # 使用噪声信号作为参考优化LMS滤波
        reference = noise  # 直接使用纯噪声作为参考
        
        filtered = audio.copy()
        if len(audio.shape) > 1:
            for channel in range(audio.shape[1]):
                filtered[:, channel] = adaptive_noise_cancellation(
                    audio[:, channel],
                    reference,
                    filter_length,
                    mu
                )
        else:
            filtered = adaptive_noise_cancellation(audio, reference, filter_length, mu)
    
    # 保存处理后的音频
    sf.write(output_path, filtered, sr)
    return True

def adaptive_noise_cancellation(primary, reference, filter_length, mu):
    """改进的自适应噪声消除器"""
    w = np.zeros(filter_length)
    y = np.zeros_like(primary)
    
    # 分块处理以提高效率
    block_size = 2048
    overlap = filter_length
    
    for i in range(0, len(primary), block_size - overlap):
        block_end = min(i + block_size, len(primary))
        
        # 获取当前数据块
        d = primary[i:block_end]
        x = reference[i:block_end]
        
        # 对当前块进行处理
        for n in range(filter_length, len(d)):
            x_buf = x[n-filter_length:n][::-1]
            y[i+n] = np.dot(w, x_buf)
            e = d[n] - y[i+n]
            
            # 归一化LMS更新
            power = np.dot(x_buf, x_buf) + 1e-10
            w = w + 2 * mu * e * x_buf / power
    
    return primary - y


# 示例调用
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="音频降噪：IIR 陷波 / LMS 自适应滤波")
    parser.add_argument('input_file', help='输入 WAV 文件路径')
    parser.add_argument('output_file', help='输出 WAV 文件路径')
    parser.add_argument('noise_file', help='纯噪声 WAV 文件路径')
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
        noise_path=args.noise_file,
        filter_type=args.filter,
        noise_params={'frequency': args.f0, 'type': 'steady' if args.filter == 'iir' else 'non_steady'},
        Q=args.Q,
        filter_length=args.length,
        mu=args.mu
    )
    print(f"输出文件: {result}")
