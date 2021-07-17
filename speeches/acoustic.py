from typing import Optional, Tuple

import tensorflow as tf

from .speechset import SpeechSet
from ..config import Config
from ..datasets.reader import DataReader
from ..utils.melstft import MelSTFT
from ..utils.normalizer import TextNormalizer


class AcousticDataset(SpeechSet):
    """Dataset for text to acoustic features.
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
        self.melstft = MelSTFT(config)
        self.textnorm = TextNormalizer()

    def reader(self) -> DataReader:
        """Get file-format datum reader.
        Returns:
            data reader.
        """
        return self.rawset
    
    def _norm_datum(self, text: tf.Tensor, speech: tf.Tensor) \
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
        labels = self.textnorm.tf_labeler(text)
        # S
        textlen = tf.shape(labels)[0]
        # [T // hop, mel]
        mel = tf.squeeze(self.melstft(speech[None]), axis=0)
        # T // hop
        mellen = tf.shape(mel)[0]
        return labels, mel, textlen, mellen

    def normalize(self, rawset: tf.data.Dataset) -> tf.data.Dataset:
        """Compose preprocessor.
        Args:
            rawset: raw dataset, expected format
                text: tf.string, text.
                speech: [tf.float32; T], speech signal in range (-1, 1).
        Returns:
            preprocessed dataset.
                text: [tf.int32; [B, S]], labeled sequence.
                mel: [tf.float32; [B, T // hop, config.mel]], mel-spectrogram.
                textlen: [tf.int32; [B]], text lengths.
                mellen: [tf.int32; [B]], mel lengths.
        """
        dataset = rawset.map(self._norm_datum)
        if self.config.batch is not None:
            dataset = dataset.padded_batch(
                self.config.batch,
                padded_shapes=([None], [None, self.config.mel], [], []))
        return dataset
