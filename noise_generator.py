import numpy as np
from scipy import signal

def generate_steady_noise(length, sr, freq, amplitude):
    """生成稳态噪声"""
    t = np.arange(length) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)

def generate_non_steady_noise(length, sr, freq, amplitude):
    """生成非稳态噪声
    
    修改策略：
    1. 增加突发噪声的强度
    2. 改进突发点的选择方式
    3. 增加多种噪声类型的叠加
    4. 调整突发持续时间
    """
    noise = np.zeros(length)
    t = np.arange(length) / sr
    
    # 生成突发点（每隔0.2-0.5秒随机出现）
    min_interval = int(0.2 * sr)
    max_interval = int(0.5 * sr)
    current_point = 0
    burst_points = []
    while current_point < length:
        burst_points.append(current_point)
        current_point += np.random.randint(min_interval, max_interval)
    
    # 为每个突发点生成噪声
    for point in burst_points:
        # 突发持续时间（50-150ms）
        burst_length = np.random.randint(int(0.05*sr), int(0.15*sr))
        if point + burst_length > length:
            burst_length = length - point
            
        # 突发噪声强度（原始振幅的1-2倍）
        local_amp = amplitude * (1.5 + np.random.random() * 0.5)
        
        # 生成混合突发噪声
        burst = np.zeros(burst_length)
        
        # 1. 添加高斯白噪声
        burst += np.random.normal(0, local_amp * 0.5, burst_length)
        
        # 2. 添加调频信号
        t_burst = np.arange(burst_length) / sr
        f_start = freq * (0.8 + np.random.random() * 0.4)
        f_end = freq * (1.2 + np.random.random() * 0.4)
        chirp = local_amp * signal.chirp(t_burst, f_start, t_burst[-1], f_end)
        burst += chirp
        
        # 3. 添加冲击响应
        impulse_pos = np.random.randint(0, burst_length)
        impulse = local_amp * 2 * np.exp(-np.arange(burst_length-impulse_pos)/(0.01*sr))
        burst[impulse_pos:] += impulse
        
        # 应用突发包络（快速起始，缓慢衰减）
        attack = int(0.1 * burst_length)
        envelope = np.ones(burst_length)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack:] *= np.exp(-np.arange(burst_length-attack)/(0.05*sr))
        
        burst *= envelope
        
        # 添加到主噪声信号
        if point + burst_length <= length:
            noise[point:point+burst_length] += burst
    
    # 添加小幅背景噪声
    background_noise = np.random.normal(0, amplitude * 0.1, length)
    noise += background_noise
    
    return noise

def add_noise_to_audio(audio_data, noise_type, freq, amplitude, sr):
    """
    将噪声添加到音频信号中
    """
    if noise_type == "steady":
        noise = generate_steady_noise(len(audio_data), sr, freq, amplitude)
    else:
        noise = generate_non_steady_noise(len(audio_data), sr, freq, amplitude)
    
    # 归一化处理，避免溢出
    max_amplitude = np.max(np.abs(audio_data))
    if max_amplitude > 0:
        noise = noise * max_amplitude
    
    return audio_data + noise
