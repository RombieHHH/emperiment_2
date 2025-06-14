import os
import sys
import json
import librosa
import numpy as np
from openai import OpenAI
from scipy.stats import entropy as scipy_entropy


# 自动获取 API 密钥（兼容 OPENAI_API_KEY 和 DEEPSEEK_API_KEY）
API_KEY = "sk-2b4d43c23f4a4e45acc5a3be24bc14c9"
if not API_KEY:
    sys.exit("请设置环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")

BASE_URL = "https://api.deepseek.com"

# 初始化 Deepseek 客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 计算谱通量
def spectral_flux(magnitude: np.ndarray) -> np.ndarray:
    flux = np.sqrt(np.sum((np.diff(magnitude, axis=1))**2, axis=0))
    return np.concatenate(([0], flux))

# 计算谱熵
def spectral_entropy(magnitude: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    ps = magnitude + eps
    ps_norm = ps / np.sum(ps, axis=0, keepdims=True)
    ent = scipy_entropy(ps_norm, base=2, axis=0)
    return ent

# 提取 STFT 频谱特征
def extract_spectral_features(y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> dict:
    S = np.abs(librosa.stft(
        y,
        n_fft=frame_length * 2,
        hop_length=hop_length,
        win_length=frame_length
    ))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    flux = spectral_flux(S)
    entropy = spectral_entropy(S)
    features = {
        "centroid_mean": float(np.mean(centroid)),
        "centroid_var": float(np.var(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "bandwidth_var": float(np.var(bandwidth)),
        "flux_mean": float(np.mean(flux)),
        "flux_var": float(np.var(flux)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_var": float(np.var(entropy)),
    }
    return features

# 调用 Deepseek Chat API 进行分类
def classify_noise_deepseek(features: dict) -> str:
    messages = [
        {"role": "system", "content": (
            "你是一个音频信号处理专家。根据给定的 STFT 特征，判断噪声类型是'稳态噪声'还是'非稳态噪声'。"
            "请严格返回如下 JSON 格式：{\"label\": \"稳态噪声\"} 或 {\"label\": \"非稳态噪声\"}，不要添加其他内容。"
        )},
        {"role": "user", "content": json.dumps(features, ensure_ascii=False)}
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    reply = response.choices[0].message.content.strip()
    try:
        return json.loads(reply)["label"]
    except:
        return "未知"

def classify_audio(audio_path: str) -> tuple:
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = int(0.025 * sr)
    hop_length = int(0.01 * sr)
    features = extract_spectral_features(y, sr, frame_length, hop_length)
    label = classify_noise_deepseek(features)
    return features, label

def get_filter_suggestion(snr_before, snr_after, filter_type, params: dict) -> str:
    """
    调用大模型，根据 SNR 提升和滤波参数生成优化建议。
    """
    prompt = (
        f"作为音频信号处理专家，请对以下滤波结果进行分析并给出具体建议：\n"
        f"1. 当前使用的滤波器：{filter_type.upper()}\n"
        f"2. 滤波参数：\n"
        f"   - 频率(f0): {params['f0']} Hz\n"
        f"   - {'Q值: ' + str(params['Q']) if filter_type=='iir' else 'LMS步长(μ): ' + str(params['lms_mu'])}\n"
        f"   {('LMS滤波器长度: ' + str(params['lms_length'])) if filter_type=='lms' else ''}\n"
        f"3. 信噪比变化：\n"
        f"   - 处理前：{snr_before:.2f} dB\n"
        f"   - 处理后：{snr_after:.2f} dB\n"
        f"   - 提升：{snr_after-snr_before:.2f} dB\n\n"
        "请给出具体的参数优化建议，要求：\n"
        "1. 分析当前参数是否合适\n"
        "2. 给出明确的参数调整方向\n"
        "3. 简要说明调整理由\n"
        "请用中文回答，简明扼要。"
    )
    
    try:
        messages = [
            {"role": "system", "content": "你是一位专业的音频信号处理专家，擅长滤波器参数优化。"},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False,
            temperature=0.7,  # 降低随机性
            max_tokens=500    # 限制回复长度
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"获取建议失败: {str(e)}"

# CLI 支持和模块导入兼容
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="使用 Deepseek Chat API 提取 STFT 特征并分类噪声类型"
    )
    parser.add_argument('audio_path', type=str, help='输入 WAV 文件路径')
    args = parser.parse_args()
    feats, res = classify_audio(args.audio_path)
    print(f"音频文件: {args.audio_path}")
    print("提取特征:", json.dumps(feats, ensure_ascii=False, indent=2))
    print("分类结果:", res)