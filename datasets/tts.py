from typing import Optional, Tuple

import librosa
import tensorflow as tf

from .config import Config
from .normalizer import TextNormalizer
from .reader import DataReader


class TTSDataset:
    """TTS-Dataset.
    """
    VOCABS = len(TextNormalizer.GRAPHEMES) + 1

    def __init__(self, rawset: DataReader, config: Config):
        """Initializer.
        Args:
            rawset: file-format datum reader.
            config: configuration.
        """
        self.rawset = rawset
        self.config = config
        self.normalized = None
        # [fft // 2 + 1, mel]
        melfilter = librosa.filters.mel(
            config.sr, config.fft, config.mel, config.fmin, config.fmax).T
        self.melfilter = tf.convert_to_tensor(melfilter)
        self.textnorm = TextNormalizer()

    def labeler(self, text: tf.Tensor) -> tf.Tensor:
        """Convert text to integer label.
        Args:
            text: string, text.
        Returns:
            labels: [tf.int32; S], labels.
        """
        text = text.numpy().decode('utf-8')
        labels = self.textnorm.labeling(text)
        return tf.convert_to_tensor(labels, dtype=tf.int32)

    def normalize(self, text: tf.Tensor, speech: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Normalize datum.
        Args:
            text: tf.string, text.
            speech: [tf.float32; T], speech in range (-1, 1).
        Returns:
            normalized datum.
                labels: [tf.int32; S], labeled text sequence.
                mel: [tf.float32; [T // hop, mel]], mel spectrogram.
                textlen: tf.int32, text lengths.
                mellen: tf.int32, mel lengths.
        """
        # [S]
        labels = tf.py_function(self.labeler, [text], tf.int32)
        # S
        textlen = tf.shape(labels)[0]
        # [T // hop, mel]
        mel = tf.squeeze(self.mel_fn(speech[None]), axis=0)
        # T // hop
        mellen = tf.shape(mel)[0]
        return labels, mel, textlen, mellen

    def mel_fn(self, signal: tf.Tensor) -> tf.Tensor:
        """Generate log mel-spectrogram from input audio segment.
        Args:
            signal: [tf.float32; [B, T]], audio segment.
        Returns:
            logmel: [tf.float32; [B, T // hop, mel]], log mel-spectrogram.
        """
        padlen = self.config.win // 2
        # [B, T + win - 1]
        center_pad = tf.pad(signal, [[0, 0], [padlen, padlen]], mode='reflect')
        # [B, T // hop, fft // 2 + 1]
        stft = tf.signal.stft(
            center_pad,
            frame_length=self.config.win,
            frame_step=self.config.hop,
            fft_length=self.config.fft,
            window_fn=self.config.window_fn())
        # [B, T // hop, mel]
        mel = tf.abs(stft) @ self.melfilter
        # [B, T // hop, mel]
        logmel = tf.math.log(tf.maximum(mel, self.config.eps))
        return logmel

    def preproc(self, rawset: tf.data.Dataset) -> tf.data.Dataset:
        """Compose preprocessor.
        Args:
            rawset: raw dataset, expected format
                text: tf.string, text.
                speech: [tf.float32; T], speech signal in range (-1, 1).
        Returns:
            preprocessed dataset.
                text: [tf.int32; [B, S]], labeled sequence.
                mel: [tf.float32; [B, T, config.mel]], mel-spectrogram.
                textlen: [tf.int32; [B]], text lengths.
                mellen: [tf.int32; [B]], mel lengths.
        """
        return rawset \
            .map(self.normalize) \
            .padded_batch(
                self.config.batch,
                padded_shapes=([None], [None, self.config.mel], [], []))

    def dataset(self, split: Optional[int] = None) \
            -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        """Generate dataset.
        Args:
            split: train-test split point, size of the training samples.
                if none is given, test set would not be provided.
        Returns:
            training, test dataset.
                text: [tf.int32; [B, S]], labeled sequence.
                mel: [tf.float32; [B, T, config.mel]], mel-spectrogram.
                textlen: [tf.int32; [B]], text lengths.
                mellen: [tf.int32; [B]], mel lengths.
        """
        dataset, preproc = self.rawset.dataset(), self.rawset.preproc()
        if split is None:
            return self.preproc(dataset.map(preproc)), None
        # split and preprocess
        train = self.preproc(dataset.take(split).map(preproc))
        test = self.preproc(dataset.skip(split).map(preproc))
        return train, test
