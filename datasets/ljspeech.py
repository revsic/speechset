import os
from typing import Callable, Dict, List, Tuple

import librosa
import numpy as np

from .reader import DataReader


class LJSpeech(DataReader):
    """LJ Speech dataset loader.
    Use other opensource vocoder settings, 16bit, sr: 22050.
    """
    SR = 22050
    MAXVAL = 32767.

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

    def load_data(self, data_dir: str) -> Tuple[List[str], Callable]:
        """Load audio with tf apis.
        Args:
            data_dir: dataset directory.
        Returns:
            data loader.
                text: str, text.
                speech: [np.float32; T], speech signal in range (-1, 1).
        """
        # generate file lists
        files = [
            os.path.join(data_dir, 'wavs', filename)
            for filename in os.listdir(os.path.join(data_dir, 'wavs'))
            if filename.endswith('.wav')]
        # read filename-text pair
        with open(os.path.join(data_dir, 'metadata.csv'), encoding='utf-8') as f:
            table = {}
            for row in f.readlines():
                name, _, normalized = row.replace('\n', '').split('|')
                table[name] = normalized
        # read audio
        return files, table

    def preprocessor(self, path: str) -> Tuple[str, np.ndarray]:
        """Load audio and lookup text.
        Args:
            path: str, path
        Returns:
            tuple,
                text: str, text.
                audio: [np.float32; T], raw speech signal in range(-1, 1).
        """
        # [T]
        audio, _ = librosa.load(path, sr=LJSpeech.SR)
        # str
        path = os.path.basename(path).replace('.wav', '')
        # str, [np.float32; T]
        return self.transcript.get(path, ''), audio.astype(np.float32)
