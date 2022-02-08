import os
from typing import Callable, List, Tuple

import librosa
import numpy as np

from .reader import DataReader


class LibriTTS(DataReader):
    """LibriTTS dataset loader.
    Use other opensource settings, 16bit, sr: 22050 (originally 24khz).
    """
    SR = 22050

    def __init__(self, data_dir: str):
        """Initializer.
        Args:
            data_dir: dataset directory.
        """
        self.filelist, self.transcript = self.load_data(data_dir)

    def dataset(self) -> List[str]:
        """Return file reader.
        Returns:
            file-format datum reader.
        """
        return self.filelist
    
    def preproc(self) -> Callable:
        """Return data preprocessor.
        Returns:
            preprocessor, expected format 
                text: str, text.
                speech: [np.float32; T], speech signal in range (-1, 1).
        """
        return self.preprocessor

    @staticmethod
    def count_speakers(data_dir: str) -> int:
        """Count the number of speakers.
        Args:
            data_dir: target dataset directory.
        Returns:
            the number of the speakers.
        """
        return len(os.listdir(data_dir))

    def load_data(self, data_dir: str) -> Tuple[List[str], Callable]:
        """Load audio with tf apis.
        Args:
            data_dir: dataset directory.
        Returns:
            data loader.
                sid: int, speaker id.
                text: str, text.
                speech: [np.float32; T], speech signal in range (-1, 1).
        """
        # generate file lists
        paths, trans = [], {}
        for sid, speakers in enumerate(os.listdir(data_dir)):
            for chapters in os.listdir(os.path.join(data_dir, speakers)):
                path = os.path.join(data_dir, speakers, chapters)
                # read transcription
                with open(os.path.join(path, f'{speakers}_{chapters}.trans.tsv')) as f:
                    for row in f.readlines():
                        filename, _, normalized = row.replace('\n', '').split('\t')
                        trans[filename] = (sid, normalized)
                # wav files
                paths.extend([
                    os.path.join(path, filename)
                    for filename in os.listdir(path) if filename.endswith('.wav')])
        # read audio
        return paths, trans

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
        # [T]
        audio, _ = librosa.load(path, sr=LibriTTS.SR)
        # str
        path = os.path.basename(path).replace('.wav', '')
        # int, str
        sid, text = self.transcript.get(path, (-1, ''))
        # int, str, [np.float32; T]
        return sid, text, audio.astype(np.float32)
