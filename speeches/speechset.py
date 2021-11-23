from copy import deepcopy
from typing import Any, List

import numpy as np

from ..datasets import DataReader


class SpeechSet:
    """Abstraction of speech dataset.
    """
    def __init__(self, reader: DataReader):
        """Caching dataset and preprocessor from reader.
        """
        self.reader = reader
        self.dataset, self.preproc = reader.dataset(), reader.preproc()

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

    def split(self, size: int):
        """Split dataset.
        WARNING: safety of this method is guaranteed by `copy.deepcopy`.
        Args:
            size: size of the first part.
        Returns:
            residual dataset.
        """
        residual = deepcopy(self)
        residual.dataset = residual.dataset[size:]
        self.dataset = self.dataset[:size]
        return residual

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

    def __iter__(self):
        """Construct iterator.
        Returns:
            SpeechSet.Iterator, index-based iterator.
        """
        return SpeechSet.Iterator(self)

    def __len__(self) -> int:
        """Return length of the dataset.
        Returns:
            length.
        """
        return len(self.dataset)

    class Iterator:
        """Index-based iterator.
        """
        def __init__(self, speechset):
            """Initializer.
            Args:
                speechset: SpeechSet, dataset.
            """
            self.speechset = speechset
            self.index = 0
        
        def __next__(self) -> Any:
            """Sampling.
            Returns:
                normalized data.
            """
            if self.index >= len(self.speechset):
                raise StopIteration
            # sampling
            datum = self.speechset[self.index]
            # successor
            self.index += 1
            return datum
