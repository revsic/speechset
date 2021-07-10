from typing import Optional, Tuple

import tensorflow as tf

from ..datasets import DataReader


class SpeechSet:
    """Abstraction of speech dataset.
    """
    def reader(self) -> DataReader:
        """Get file-format data reader.
        Returns:
            data reader.
        """
        raise NotImplementedError('SpeechSet.reader is not implemented')

    def normalize(self, rawset: tf.data.Dataset) -> tf.data.Dataset:
        """Normalizer.
        Args:
            rawset: file-format raw dataset.
        Returns:
            normalized batch-level dataset.
        """
        raise NotImplementedError('SpeechSet.preproc is not implemented')
    
    def dataset(self, split: Optional[int] = None) \
            -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        """Generate dataset.
        Args:
            split: train-text split point, size of the training samples.
                if none is given, test set would not be provided.
        Returns:
            training, test dataset.
        """
        reader = self.reader()
        dataset, preprocessor = reader.dataset(), reader.preproc()
        if split is None:
            return self.normalize(dataset.map(preprocessor)), None
        # split and preprocess
        train = self.normalize(dataset.take(split).map(preprocessor))
        test = self.normalize(dataset.skip(split).map(preprocessor))
        return train, test
