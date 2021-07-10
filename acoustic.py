from typing import Optional, Tuple

import tensorflow as tf

from .config import Config
from .datasets.reader import DataReader
from .utils.melstft import MelSTFT
from .utils.normalizer import TextNormalizer


class AcousticDataset:
    """Datset for text to acoustic features.
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
        labels = self.textnorm.tf_labeler(text)
        # S
        textlen = tf.shape(labels)[0]
        # [T // hop, mel]
        mel = tf.squeeze(self.melstft(speech[None]), axis=0)
        # T // hop
        mellen = tf.shape(mel)[0]
        return labels, mel, textlen, mellen

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
