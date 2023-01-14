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

    def speakers(self) -> List[str]:
        """Return list of speakers.
        Returns:
            list of the speakers.
        """
        raise NotImplementedError('DataReader.speakers is not implemented')

    def preproc(self) -> Callable:
        """Return data preprocessor.
        Returns:
            preprocessor, required format,
                (optional) sid: int, speaker id.
                text: str, text.
                speech: [np.float32; T], speech signal in range (-1, 1).
        """
        raise self._preproc_template

    def _preproc_template(self, path: str) -> Tuple[int, str, np.ndarray]:
        """Load audio and lookup text.
        Args:
            path: str, path
        Returns:
            tuple,
                sid: int, speaker id.
                text: str, text.
                audio: [np.float32; T], raw speech signal in range(-1, 1).
        """
        trans = self.dataset()
        # [T]
        audio = self.load_audio(path, self.sr)
        # int, str
        sid, text = trans.get(path, (-1, ''))
        # int, str, [np.float32; T]
        return sid, text, audio
