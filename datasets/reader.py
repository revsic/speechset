from typing import Callable, Dict, List, Tuple

import librosa
import numpy as np


class DataReader:
    """Interface of the data reader for efficient train-test split.
    """
    def load_audio(self, path: str, sr: int) -> np.ndarray:
        """Read the audio.
        Args:
            path: path to the audio.
            sr: sampling rate.
        Returns:
            [np.float32; [T]], audio signal, [-1, 1]-ranged.
        """
        audio, _ = librosa.load(path, sr=sr)
        return audio.astype(np.float32)

    def dataset(self) -> Dict[str, Tuple[int, str]]:
        """Return file reader.
        Returns:
            file-format datum reader, without any preprocessor for fast train-text split.
        """
        raise NotImplementedError('DataReader.rawset is not implemented')

    def preproc(self) -> Callable:
        """Return data preprocessor.
        Returns:
            preprocessor, required format,
                (optional) sid: int, speaker id.
                text: str, text.
                speech: [np.float32; T], speech signal in range (-1, 1).
        """
        raise NotImplementedError('DataReader.preproc is not implemented')

    def speakers(self) -> List[str]:
        """Return list of speakers.
        Returns:
            list of the speakers.
        """
        raise NotImplementedError('DataReader.speakers is not implemented')
