import numpy as np

def generate_steady_noise(length, sr, freq, amplitude):
    """生成稳态噪声"""
    t = np.arange(length) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)

def generate_non_steady_noise(length, sr, freq, amplitude):
    """生成非稳态噪声"""
    t = np.arange(length) / sr
    # 这里使用调幅正弦波作为非稳态噪声的示例
    carrier = np.sin(2 * np.pi * freq * t)
    modulator = 1 + 0.5 * np.sin(2 * np.pi * 2 * t)
    return amplitude * carrier * modulator

def add_noise_to_audio(audio_data, noise_type, freq, amplitude, sr):
    """
    将噪声添加到音频信号中
    
    Parameters:
    -----------
    audio_data : np.ndarray
        输入音频数据
    noise_type : str
        噪声类型 ('steady' 或 'non_steady')
    freq : float
        噪声基频
    amplitude : float
        噪声振幅
    sr : int
        采样率
    
    Returns:
    --------
    np.ndarray
        添加噪声后的音频数据
    """
    if noise_type == "steady":
        noise = generate_steady_noise(len(audio_data), sr, freq, amplitude)
    else:
        noise = generate_non_steady_noise(len(audio_data), sr, freq, amplitude)
    
    return audio_data + noise
