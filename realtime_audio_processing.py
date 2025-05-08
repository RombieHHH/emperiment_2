import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio
import librosa
import librosa.display
from filter import AudioFilter

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000

class RealTimeAudioProcessor:
    def __init__(self, filter_type='iir', f0=50.0, Q=30.0, filter_length=32, mu=0.01):
        self.filter_type = filter_type
        self.f0 = f0
        self.Q = Q
        self.filter_length = filter_length
        self.mu = mu
        self.audio_filter = AudioFilter(filter_length=filter_length, mu=mu)
        self.frames = []
        self.filtered_frames = []
        self.snr_curve = []

    def process_chunk(self, chunk):
        data = np.frombuffer(chunk, dtype=np.float32)
        if self.filter_type == 'iir':
            filtered = self.audio_filter.iir_notch_filter(data, RATE, self.f0, self.Q)
        else:
            ref = self.audio_filter.generate_reference(len(data), RATE, self.f0)
            filtered = self.audio_filter.lms_filter(data, ref, self.mu)
        return filtered

    def compute_snr(self, original, processed):
        noise = original - processed
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return snr

    def start(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        def update(frame):
            data = stream.read(CHUNK, exception_on_overflow=False)
            y = np.frombuffer(data, dtype=np.float32)
            y_filt = self.process_chunk(data)
            self.frames.append(y)
            self.filtered_frames.append(y_filt)
            if len(self.frames) > 50:
                self.frames.pop(0)
                self.filtered_frames.pop(0)
            # 波形
            axs[0].cla()
            axs[0].plot(np.concatenate(self.frames), label='原始')
            axs[0].plot(np.concatenate(self.filtered_frames), label='滤波')
            axs[0].legend()
            axs[0].set_title("实时波形")
            # 语谱图
            axs[1].cla()
            D = librosa.amplitude_to_db(np.abs(librosa.stft(np.concatenate(self.filtered_frames))), ref=np.max)
            librosa.display.specshow(D, sr=RATE, x_axis='time', y_axis='hz', ax=axs[1])
            axs[1].set_title("滤波后语谱图")
            # SNR 曲线
            axs[2].cla()
            if len(self.frames) > 0:
                snr = self.compute_snr(np.concatenate(self.frames), np.concatenate(self.filtered_frames))
                self.snr_curve.append(snr)
                if len(self.snr_curve) > 100:
                    self.snr_curve.pop(0)
                axs[2].plot(self.snr_curve)
            axs[2].set_title("SNR 曲线")
            plt.tight_layout()

        ani = animation.FuncAnimation(fig, update, interval=50)
        plt.show()
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    processor = RealTimeAudioProcessor(filter_type='iir')
    processor.start()
