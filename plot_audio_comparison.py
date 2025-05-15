import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

def compute_snr(original, processed):
    """
    简单 SNR 计算：原始信号视为信号，误差信号视为噪声
    """
    noise = original - processed
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-12:
        return np.nan  # 理论上无穷大
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr

def plot_comparison(original_path: str, processed_path: str, clean_path: str = None, title_prefix=""):
    plt.close('all')  # 关闭所有旧窗口

    # 读取音频
    y_orig, sr_orig = sf.read(original_path)  # 含噪音频
    y_proc, sr_proc = sf.read(processed_path)  # 处理后音频
    if clean_path:
        y_clean, sr_clean = sf.read(clean_path)  # 干净音频
        if sr_clean != sr_orig:
            raise ValueError("干净音频采样率与其他音频不一致")
    else:
        y_clean = None
    
    if sr_orig != sr_proc:
        raise ValueError("原始音频和处理后音频采样率不一致")
    
    # 转换为单声道
    if y_orig.ndim == 2:
        y_orig = y_orig.mean(axis=1)
    if y_proc.ndim == 2:
        y_proc = y_proc.mean(axis=1)
    if y_clean is not None and y_clean.ndim == 2:
        y_clean = y_clean.mean(axis=1)

    # 确保长度一致
    min_len = min(len(y_orig), len(y_proc))
    if y_clean is not None:
        min_len = min(min_len, len(y_clean))
    y_orig = y_orig[:min_len]
    y_proc = y_proc[:min_len]
    if y_clean is not None:
        y_clean = y_clean[:min_len]

    font = {'family': 'Microsoft YaHei', 'size': 13}

    # 创建三个子图：波形、语谱图和SNR曲线
    fig = plt.figure(figsize=(16, 12))
    
    # 波形对比
    ax1 = plt.subplot(3, 1, 1)
    # if y_clean is not None:
    #     plt.plot(y_clean, label='原始信号', alpha=0.7, color='green')
    plt.plot(y_orig, label='含噪信号', alpha=0.7, color='red')
    plt.plot(y_proc, label='滤波后信号', alpha=0.7, color='blue')
    plt.title(f"{title_prefix}波形对比", fontdict=font)
    plt.legend(prop=font)
    plt.grid(True)

    # 语谱图对比
    ax2 = plt.subplot(3, 1, 2)
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig)), ref=np.max)
    D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_proc)), ref=np.max)
    plt.subplot(3, 2, 3)
    librosa.display.specshow(D_orig, sr=sr_orig, x_axis='time', y_axis='hz')
    plt.title(f"{title_prefix}含噪信号语谱图", fontdict=font)
    plt.colorbar(format="%+2.0f dB")
    plt.subplot(3, 2, 4)
    librosa.display.specshow(D_proc, sr=sr_orig, x_axis='time', y_axis='hz')
    plt.title(f"{title_prefix}滤波后语谱图", fontdict=font)
    plt.colorbar(format="%+2.0f dB")

    # SNR曲线对比
    if y_clean is not None:
        ax3 = plt.subplot(3, 1, 3)
        # 计算分段SNR
        window_size = int(0.05 * sr_orig)  # 50ms窗口
        snr_noisy = []
        snr_filtered = []
        
        for i in range(0, min_len - window_size, window_size // 2):
            clean_segment = y_clean[i:i+window_size]
            orig_segment = y_orig[i:i+window_size]
            proc_segment = y_proc[i:i+window_size]
            
            # 计算含噪信号的SNR
            noise_orig = orig_segment - clean_segment
            snr_noisy.append(10 * np.log10(np.mean(clean_segment**2) / (np.mean(noise_orig**2) + 1e-10)))
            
            # 计算滤波后信号的SNR
            noise_proc = proc_segment - clean_segment
            snr_filtered.append(10 * np.log10(np.mean(clean_segment**2) / (np.mean(noise_proc**2) + 1e-10)))

        time_points = np.arange(len(snr_noisy)) * (window_size/2) / sr_orig
        plt.plot(time_points, snr_noisy, label='含噪信号SNR', color='red')
        plt.plot(time_points, snr_filtered, label='滤波后SNR', color='blue')
        plt.title("SNR对比", fontdict=font)
        plt.xlabel("时间 (秒)", fontdict=font)
        plt.ylabel("SNR (dB)", fontdict=font)
        plt.legend(prop=font)
        plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close('all')
    
    # 返回平均SNR值
    if y_clean is not None:
        return (np.mean(snr_noisy), np.mean(snr_filtered))
    return None
