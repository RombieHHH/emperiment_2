import numpy as np
from scipy import signal

def generate_noise(type, frequency, amplitude, sample_rate, duration, **kwargs):
    """生成指定类型的噪声信号"""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, endpoint=False)
    
    if type == 'steady':
        # 生成稳态正弦噪声
        noise = amplitude * np.sin(2 * np.pi * frequency * t)
        # 添加谐波分量
        for i in range(2, 4):
            noise += (amplitude / i) * np.sin(2 * np.pi * (frequency * i) * t)
            
    else:  # non_steady
        # 生成非稳态噪声
        noise = np.zeros(samples)
        # 添加时变频率成分
        mod_freq = frequency * 0.1  # 调制频率
        # 使用调频信号
        inst_freq = frequency * (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t))
        phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
        noise = amplitude * np.sin(phase)
        # 添加随机振幅调制
        env = 1 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
        noise *= env
        
    return noise

def add_noise_to_audio(audio, noise_type, freq, amp, sr):
    """将噪声添加到音频信号"""
    duration = len(audio) / sr
    noise = generate_noise(noise_type, freq, amp, sr, duration)
    if len(audio.shape) > 1:  # 处理立体声
        noise = np.column_stack((noise, noise))
    return audio + noise
