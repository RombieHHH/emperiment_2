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

    def lms_filter(self, data: np.ndarray, reference: np.ndarray, mu: float = None) -> np.ndarray:
        """
        对输入信号应用 LMS 自适应滤波。

        Args:
            data: 输入信号（1D 或 2D 立体声）。
            reference: 参考噪声信号，应与 data 等长。
            mu: 可选的步长因子，默认使用实例属性 self.mu。

        Returns:
            滤波后信号，与输入形状相同。
        """
        if mu is not None:
            self.mu = mu

        # 判断是否为立体声
        if data.ndim == 2:
            left = self._lms_process_channel(data[:, 0], reference)
            right = self._lms_process_channel(data[:, 1], reference)
            return np.stack((left, right), axis=1)
        else:
            return self._lms_process_channel(data, reference)

    def _lms_process_channel(self, channel: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        对单声道信号执行 LMS 运算。
        """
        # 初始化权重和输出
        w = np.zeros(self.filter_length)
        y = np.zeros_like(channel)

        # 从 filter_length 开始计算
        for n in range(self.filter_length, len(channel)):
            x_vec = reference[n - self.filter_length:n][::-1]
            y[n] = np.dot(w, x_vec)
            e = channel[n] - y[n]
            w += 2 * self.mu * e * x_vec

        # 返回原始信号减去估计噪声
        return channel - y

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
        对输入信号应用 IIR 陷波滤波器，抑制指定频率。

        Args:
            data: 输入信号（1D 或 2D 立体声）。
            fs: 采样率（Hz）。
            f0: 陷波频率，默认为 50Hz。
            Q: 品质因数，值越高带阻越窄，默认为 5。

        Returns:
            滤波后信号。与输入形状相同。
        """
        if data.ndim == 2:
            left = self._iir_process_channel(data[:, 0], fs, f0, Q)
            right = self._iir_process_channel(data[:, 1], fs, f0, Q)
            return np.stack((left, right), axis=1)
        else:
            return self._iir_process_channel(data, fs, f0, Q)

    def _iir_process_channel(self, channel: np.ndarray, fs: int, f0: float, Q: float) -> np.ndarray:
        """
        IIR 单声道处理，优先使用 filtfilt，失败时退化到 lfilter。
        """
        # 归一化频率
        w0 = f0 / (fs / 2)
        b, a = signal.iirnotch(w0, Q)
        try:
            return signal.filtfilt(b, a, channel)
        except ValueError:
            return signal.lfilter(b, a, channel)


# 全局函数：批量处理并保存结果

def process_audio(input_path: str, filter_type: str = 'iir',
                  f0: float = 50.0, Q: float = 30.0,
                  filter_length: int = 32, mu: float = 0.01,
                  output_dir: str = 'output') -> str:
    """
    读取音频文件，应用指定滤波器并保存结果。

    Args:
        input_path: 输入 WAV 文件路径。
        filter_type: 'iir' 或 'lms'。
        f0: 陷波中心频率，用于 IIR。
        Q: IIR 品质因数。
        filter_length: LMS 滤波器长度。
        mu: LMS 学习率。
        output_dir: 输出目录。

    Returns:
        输出文件路径，若失败返回 None。
    """
    try:
        # 确保输出目录存在
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 读取音频
        data, fs = sf.read(input_path)
        if data.size == 0:
            raise ValueError("音频文件为空")

        # 初始化滤波器
        audio_filter = AudioFilter(filter_length=filter_length, mu=mu)

        # 选择滤波类型
        if filter_type.lower() == 'lms':
            ref = audio_filter.generate_reference(len(data), fs, f0)
            filtered = audio_filter.lms_filter(data, ref)
            suffix = 'LMS'
        else:
            filtered = audio_filter.iir_notch_filter(data, fs, f0, Q)
            suffix = 'IIR'

        # 构造输出文件名
        stem = Path(input_path).stem
        out_path = out_dir / f"{stem}_{suffix}.wav"
        sf.write(str(out_path), filtered, fs)
        return str(out_path)

    except Exception as e:
        print(f"处理音频文件时出错: {e}")
        return None


# 示例调用
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="音频降噪：IIR 陷波 / LMS 自适应滤波")
    parser.add_argument('input_file', help='输入 WAV 文件路径')
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
    parser.add_argument('--out', default='output',
                        help='输出目录')
    args = parser.parse_args()

    result = process_audio(
        input_path=args.input_file,
        filter_type=args.filter,
        f0=args.f0,
        Q=args.Q,
        filter_length=args.length,
        mu=args.mu,
        output_dir=args.out
    )
    print(f"输出文件: {result}")
