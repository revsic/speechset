import os
from typing import Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np

from .reader import DataReader


class LJSpeech(DataReader):
    """LJ Speech dataset loader.
    Use other opensource vocoder settings, 16bit, sr: 22050.
    """
    SR = 22050

    def __init__(self, data_dir: str, sr: Optional[int] = None):
        """Initializer.
        Args:
            data_dir: dataset directory.
            sr: sampling rate.
        """
        self.sr = sr or LJSpeech.SR
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

    def speakers(self) -> List[str]:
        """List of speakers.
        Returns:
            list of the speakers.
        """
        return ['ljspeech']

    def load_data(self, data_dir: str) -> Tuple[List[str], Dict[str, str]]:
        """Load audio with tf apis.
        Args:
            data_dir: dataset directory.
        Returns:
            list of audio files and transcript table.
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
        audio = self.load_audio(path, self.sr)
        # str
        path = os.path.basename(path).replace('.wav', '')
        # str, [np.float32; T]
        return self.transcript.get(path, ''), audio
