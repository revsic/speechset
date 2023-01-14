from typing import Callable, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .reader import DataReader


class ConcatReader(DataReader):
    """Concatenated data reader.
    """
    def __init__(self, readers: List[DataReader]):
        """Initializer.
        Args:
            readers: list of data readers.
        """
        self.readers = readers
        self.speakers_ = [reader.speakers() for reader in readers]
        # compute starting indices
        indices = np.cumsum([0] + [len(speakers) for speakers in self.speakers_])
        self.transcript = {
            path: (sid + start, transcript)
            for reader, start in zip(tqdm(self.readers, desc='concat'), indices)
            for path, (sid, transcript) in tqdm(reader.dataset().items(), leave=False)}
        # caching processor
        self.mapper = {
            path: reader.preproc()
            for reader in self.readers
            for path in reader.dataset()}

    def dataset(self) -> Dict[str, Tuple[int, str]]:
        """Return file reader.
        Returns:
            file-format datum reader.
        """
        return self.transcript

    def speakers(self) -> List[str]:
        """Return list of speakers.
        Returns:
            list of the speakers.
        """
        return [name
            for speakers in self.speakers_
            for name in speakers]

    def preproc(self) -> Callable:
        """Return the preprocessor.
        Returns:
            preprocessor, expected format
                sid: int, speaker id.
                text: str, text.
                audio: [np.float32; [T]], raw speech signal in range(-1, 1).
        """
        return self.preprocessor

    def preprocessor(self, path: str) -> Tuple[int, str, np.ndarray]:
        """Load audio and lookup text.
        Args:
            path: str, path
        Returns:
            tuple,
                sid: int, speaker id.
                text: str, text.
                audio: [np.float32; T], raw speech signal in range(-1, 1).
        """
        # preprocessing
        _, _, audio = self.mapper[path](path)
        # int, str
        sid, text = self.transcript.get(path, (-1, ''))
        return sid, text, audio
