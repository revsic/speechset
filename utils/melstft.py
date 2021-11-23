import librosa
import numpy as np

from ..config import Config


class MelSTFT:
    """Generate log-mel scale power spectrogram.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: STFT parameters.
        """
        self.config = config
        # [mel, fft // 2 + 1], generate mel-filters
        self.melfilter = librosa.filters.mel(
            config.sr, config.fft, config.mel, config.fmin, config.fmax)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Generate log-mel scale power spectrogram from inputs.
        Args:
            signal: [np.float32; [T]], speech signal.
        Returns:
            [np.float32; [T / hop, mel]], log-mel scale power spectrogram.
        """
        # [fft // 2 + 1, T // hop + 1]
        stft = librosa.stft(
            signal,
            self.config.fft,
            self.config.hop,
            self.config.win,
            self.config.win_fn,
            center=True, pad_mode='reflect')
        # [mel, T // hop + 1]
        mel = self.melfilter @ np.abs(stft)
        # [T // hop + 1, mel]
        return np.log(np.maximum(mel, self.config.eps)).T
