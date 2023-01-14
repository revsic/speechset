from typing import Union, List, Tuple

import numpy as np

from ..speeches.speechset import SpeechSet


class IDWrapper(SpeechSet):
    """Speechset wrapper for auxiliary ids.
    """
    def __init__(self, speechset: SpeechSet):
        """Initializer.
        Args:
            speechset: base speechset.
        """
        # hold
        self.speechset = speechset
        self.reader = speechset.reader
        # do not run higher initializer since dataset and preprocessor already cached
        self.dataset, self.preproc = speechset.dataset, speechset.preproc

    def normalize(self,
                  ids: Union[int, List[int]],
                  text: str,
                  speech: np.ndarray) \
            -> Tuple[Union[int, List[int]], Tuple[np.ndarray, np.ndarray]]:
        """Normalize datum with auxiliary ids.
        Args:
            ids: auxiliary ids.
            text: transcription.
            speech: [np.float32; [T]], speech in range (-1, 1).
        Returns:
            id and normalized datum.
        """
        return ids, self.speechset.normalize(text, speech)

    def collate(self,
                bunch: List[Tuple[Union[int, List[int]],
                            Tuple[np.ndarray, np.ndarray]]]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collate bunch of datum to the batch data.
        Args:
            bunch: B x [...], list of normalized inputs.
                ids: auxiliary ids.
                ...: normalized datum.
        Returns:
            bunch data.
                ids: [np.long; [B, ...]], auxiliary ids.
                ...: collated bunch.
        """
        # [B, ...], auxiliary ids.
        ids = self.collate_id([ids for ids, _ in bunch])
        # collated bunch
        return (ids, *self.speechset.collate([datum for _, datum in bunch]))

    def collate_id(self, bunch: List[Union[int, List[int]]]) -> np.ndarray:
        """ID collator.
        Args:
            bunch: B x [...], list of ids.
        Returns:
            [np.long; [B, ...]], collated ids.
        """
        # simple wrapping
        return np.array(bunch, dtype=np.int64)
