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

def plot_comparison(original_path: str, processed_path: str, title_prefix=""):
    plt.close('all')  # 关闭所有旧窗口

    # 读取音频
    y_orig, sr_orig = sf.read(original_path)
    y_proc, sr_proc = sf.read(processed_path)
    if sr_orig != sr_proc:
        raise ValueError("原始音频和处理后音频采样率不一致")
    if y_orig.ndim == 2:
        y_orig = y_orig.mean(axis=1)
    if y_proc.ndim == 2:
        y_proc = y_proc.mean(axis=1)
    min_len = min(len(y_orig), len(y_proc))
    y_orig = y_orig[:min_len]
    y_proc = y_proc[:min_len]

    # 分帧计算 SNR 曲线
    frame_size = int(0.05 * sr_orig)
    snr_curve_filtered = []
    for i in range(0, min_len, frame_size):
        o = y_orig[i:i+frame_size]
        f = y_proc[i:i+frame_size]
        if len(o) < frame_size:
            break
        snr_curve_filtered.append(compute_snr(o, f))

    font = {'family': 'Microsoft YaHei', 'size': 13}

    # 波形和语谱图窗口
    fig1 = plt.figure(figsize=(16, 8))

    # 波形
    plt.subplot(2, 2, 1)
    librosa.display.waveshow(y_orig, sr=sr_orig)
    plt.title(f"{title_prefix}原始波形", fontdict=font)

    plt.subplot(2, 2, 2)
    librosa.display.waveshow(y_proc, sr=sr_orig)
    plt.title(f"{title_prefix}滤波后波形", fontdict=font)

    # 语谱图
    plt.subplot(2, 2, 3)
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig)), ref=np.max)
    librosa.display.specshow(D_orig, sr=sr_orig, x_axis='time', y_axis='hz')
    plt.title(f"{title_prefix}原始语谱图", fontdict=font)
    plt.colorbar(format="%+2.0f dB")

    plt.subplot(2, 2, 4)
    D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_proc)), ref=np.max)
    librosa.display.specshow(D_proc, sr=sr_orig, x_axis='time', y_axis='hz')
    plt.title(f"{title_prefix}滤波后语谱图", fontdict=font)
    plt.colorbar(format="%+2.0f dB")

    fig1.tight_layout()

    # SNR曲线窗口
    fig2 = plt.figure(figsize=(8, 4))
    plt.plot(snr_curve_filtered, label="滤波后 SNR", color='green')
    plt.xlabel("帧", fontdict=font)
    plt.ylabel("SNR (dB)", fontdict=font)
    plt.title("SNR 曲线", fontdict=font)
    plt.legend(prop=font)
    plt.tight_layout()

    # 只调用一次plt.show()，显示所有窗口，防止卡死
    plt.show()
    plt.close('all')
