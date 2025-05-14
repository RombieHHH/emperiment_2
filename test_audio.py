import soundfile as sf
import numpy as np

def test_audio(filepath):
    y_orig, _ = sf.read(filepath)
    print(f"Audio shape: {y_orig.shape}")
    print(f"Mean amplitude: {np.mean(y_orig)}")

if __name__ == '__main__':
    test_filepath = "path/to/your/audio.wav"
    test_audio(test_filepath)
