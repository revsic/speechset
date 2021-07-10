from typing import Tuple

import tensorflow as tf

from .speechset import SpeechSet
from ..config import Config
from ..datasets.reader import DataReader
from ..utils.melstft import MelSTFT


class VocoderDataset(SpeechSet):
    """Dataset for acoustic features to audio signal.
    """
    def __init__(self, rawset: DataReader, config: Config):
        """Initializer.
        Args:
            rawset: file-format datum reader.
            config: configuration.
        """
        self.rawset = rawset
        self.config = config
        self.normalized = None
        self.melstft = MelSTFT(config)
    
    def reader(self) -> DataReader:
        """Get file-format datum reader.
        Returns:
            data reader.
        """
        return self.rawset
    
    def _datum_norm(self, _: tf.Tensor, speech: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Normalize datum.
        Args:
            _: tf.string, text placeholder.
            speech: [tf.float32; T], speech in range (-1, 1.)
        Returns:
            normalized datum.
                mel: [tf.float32; [T // hop + 1, mel]], mel spectrogram.
                speech: [tf.float32; [T]], speech signal.
                mellen: tf.int32, mel lengths.
                speechlen: tf.int32, speech lengths.
        """
        # T
        speechlen = tf.shape(speech)[0]
        # [T // hop + 1, mel]
        mel = tf.squeeze(self.melstft(speech[None]), axis=0)
        # T // hop + 1
        mellen = tf.shape(mel)[0]
        return mel, speech, mellen, speechlen

    def normalize(self, rawset: tf.data.Dataset) -> tf.data.Dataset:
        """Compose preprocessor.
        Args:
            rawset: raw dataset, expected format
                text: tf.string, text.
                speech: [tf.float32; T], speech signal in range (-1, 1).
        Returns:
            preprocessed dataset.
                mel: [tf.float32; [B, T // hop + 1, config.mel]], mel-spectrogram.
                speech: [tf.float32; [B, T]], speech signal.
                mellen: [tf.int32; [B]], mel lengths.
                speechlen: [tf.int32; [B]], speech lengths.
        """
        return rawset \
            .map(self._datum_norm) \
            .padded_batch(
                self.config.batch,
                padded_shapes=([None, self.config.mel], [None], [], []))
