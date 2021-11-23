from typing import Any, List, Tuple

import numpy as np

from ..datasets import DataReader


class SpeechSet:
    """Abstraction of speech dataset.
    """
    def __init__(self):
        """Caching dataset and preprocessor from reader.
        """
        reader = self.reader()
        self.dataset, self.preproc = reader.dataset(), reader.preproc()

    def reader(self) -> DataReader:
        """Get file-format data reader.
        Returns:
            data reader.
        """
        raise NotImplementedError('SpeechSet.reader is not implemented')

    def normalize(self, text: str, speech: np.ndarray) -> Any:
        """Normalizer.
        Args:
            text: transcription.
            speech: [np.float32; [T]], mono channel audio.
        Returns:
            normalized inputs.
        """
        raise NotImplementedError('SpeechSet.normalize is not implemented')

    def collate(self, bunch: List[Any]) -> Any:
        """Collate bunch of datum to the batch data.
        Args:
            bunch: B x [], list of normalized inputs.
        Returns:
            [B], batch data.
        """
        raise NotImplementedError('SpeechSet.collate is not implemented')

    def __getitem__(self, index: int) -> Any:
        """Lazy normalizing.
        Args:
            index: input index.
        Returns:
            normalized inputs.
        """
        # reading data
        text, speech = self.preproc(self.dataset[index])
        # normalize
        return self.normalize(text, speech)
