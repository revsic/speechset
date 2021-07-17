from typing import Callable, List

import tensorflow as tf

from .reader import DataReader


class TFRecordReader(DataReader):
    """TFRecord reader.
    """
    def __init__(self, path: str, dtypes: List[tf.dtypes.DType]):
        """Initializer.
        Args:
            path: path to the tfrecord.
            dtypes: tensor types.
        """
        super().__init__()
        self.record = tf.data.TFRecordDataset(path)
        self.dtypes = dtypes
    
    def dataset(self) -> tf.data.Dataset:
        """Return file reader.
        Returns:
            tf-record file reader.
        """
        return self.record

    def preproc(self) -> Callable:
        """Return data preprocessor.
        Returns:
            preprocessor, different from `DataReader` format,
                [self.dtypes[i]: [...]], deserialized tensors.
        """
        @tf.function
        def deserializer(inputs):
            bunch = tf.io.parse_tensor(inputs, tf.string)
            return [tf.io.parse_tensor(bunch[i], d) for i, d in enumerate(self.dtypes)]
        return deserializer
