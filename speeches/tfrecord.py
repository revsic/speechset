from typing import List, Tuple

import tensorflow as tf

from .speechset import SpeechSet
from ..config import Config
from ..datasets.tfrecord import TFRecordReader


class TFRecordDataset(SpeechSet):
    """Dataset for preprocessed tf-record.
    """
    def __init__(self, config: Config, path: str, target: str):
        """Initializer.
        Args:
            path: path to the tfrecord.
            dtypes: tensor types.
            target: target dataset.
        """
        super().__init__()
        self.config = config
        dtypes, self.padded_shapes = self.configure(target)
        self.record = TFRecordReader(path, dtypes)
    
    def reader(self) -> TFRecordReader:
        """Get tfrecord reader.
        Returns:
            tfrecord reader.
        """
        return self.record
    
    def normalize(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Compose preprocessor.
        Args:
            rawset: tfrecord dataset.
        Returns:
            batched dataset.
                [dtypes[i]; [B, ...]], bunch of data.
        """
        if self.config.batch is not None:
            dataset = dataset.padded_batch(
                self.config.batch, padded_shapes=self.padded_shapes)
        return dataset

    def configure(self, target: str) -> Tuple[List[tf.dtypes.DType], Tuple[List[int]]]:
        """Return dtypes and padded shapes.
        Args:
            target: target dataset.
        Returns:
            dtypes: data types.
            padded_shapes: padded data shapes.
        """
        assert target in ['acoustic', 'vocoder'], \
            'target should be one of the [acoustic, vocoder]'
        if target == 'acoustic':
            return \
                [tf.int32, tf.float32, tf.int32, tf.int32], \
                ([None], [None, self.config.mel], [], [])
        else:
            # vocoder
            return \
                [tf.float32, tf.float32, tf.int32, tf.int32], \
                ([None, self.config.mel], [None], [], [])
