from copy import deepcopy
from typing import Any, List, Union

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
        self.indexer = list(self.dataset.keys())

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
        residual.indexer = residual.indexer[size:]
        self.indexer = self.indexer[:size]
        return residual

    def __getitem__(self, index: Union[int, slice]) -> Any:
        """Lazy normalizing.
        Args:
            index: input index.
        Returns:
            normalized inputs.
        """
        # reading data
        raw = self.indexer[index]
        if isinstance(index, int):
            return self.normalize(*self.preproc(raw))
        # normalize for slice
        norm = [self.normalize(*self.preproc(single)) for single in raw]
        # pack
        return self.collate(norm)

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
        return len(self.indexer)

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
