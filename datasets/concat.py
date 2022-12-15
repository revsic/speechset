from typing import Callable, List, Tuple

import numpy as np

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
        self.mapper = {
            path: (reader.preproc(), start)
            for reader, start in zip(self.readers, indices)
            for path in reader.dataset()}

    def dataset(self) -> List[str]:
        """Return file reader.
        Returns:
            file-format datum reader.
        """
        return list(self.mapper.keys())

    def preproc(self) -> Callable:
        """Return the preprocessor.
        Returns:
            preprocessor, expected format
                sid: int, speaker id.
                text: str, text.
                audio: [np.float32; [T]], raw speech signal in range(-1, 1).
        """
        return self.preprocessor
    
    def speakers(self) -> List[str]:
        """Return list of speakers.
        Returns:
            list of the speakers.
        """
        return [name
            for speakers in self.speakers_
            for name in speakers]

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
        assert path in self.mapper
        preproc, start = self.mapper[path]
        
        outputs = preproc(path)
        assert len(outputs) in [2, 3]
        # without sid
        if len(outputs) == 2:
            sid = 0
            text, audio = outputs
        else:
            # with sid
            sid, text, audio = preproc(path)
        # callibrate speaker id
        return start + sid, text, audio
