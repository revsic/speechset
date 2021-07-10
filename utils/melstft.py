from typing import Optional

import librosa
import tensorflow as tf

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
        # [fft // 2 + 1, mel], generate mel-filters
        melfilter = librosa.filters.mel(
            config.sr, config.fft, config.mel, config.fmin, config.fmax).T
        self.melfilter = tf.convert_to_tensor(melfilter)

    def __call__(self, signal: tf.Tensor) -> tf.Tensor:
        """Generate log-mel scale power spectrogram from inputs.
        Args:
            signal: [tf.float32; [B, T]], speech signal.
        Returns:
            [tf.float32; [B, T // hop + 1, mel]], log-mel scale power spectrogram.
        """
        padlen = self.config.win // 2
        # [B, T + win]
        padded = tf.pad(signal, [[0, 0], [padlen, padlen]], mode='reflect')
        # [B, T // hop + 1, fft // 2 + 1]
        stft = tf.signal.stft(
            padded,
            frame_length=self.config.win,
            frame_step=self.config.hop,
            fft_length=self.config.fft,
            window_fn=self.config.window_fn())
        # [B, T // hop + 1, mel]
        mel = tf.abs(stft) @ self.melfilter
        # [B, T // hop + 1, mel]
        return tf.math.log(tf.maximum(mel, self.config.eps))
    
    def mellen(self, speechlen: tf.Tensor) -> tf.Tensor:
        """Compute length of the mel-spectrogram from source audio length.
        """
        return speechlen // self.config.hop + 1
