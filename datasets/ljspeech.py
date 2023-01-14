import os
from typing import Dict, List, Optional, Tuple

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
        self.transcript = self.load_data(data_dir)

    def dataset(self) -> Dict[str, Tuple[int, str]]:
        """Return file reader.
        Returns:
            file-format datum reader.
        """
        return self.transcript

    def speakers(self) -> List[str]:
        """List of speakers.
        Returns:
            list of the speakers.
        """
        return ['ljspeech']

    def load_data(self, data_dir: str) -> Dict[str, Tuple[int, str]]:
        """Load audio with tf apis.
        Args:
            data_dir: dataset directory.
        Returns:
            list of audio files and transcript table.
        """
        # read filename-text pair
        with open(os.path.join(data_dir, 'metadata.csv'), encoding='utf-8') as f:
            table = {}
            for row in f.readlines():
                name, _, normalized = row.replace('\n', '').split('|')
                path = os.path.join(data_dir, 'wavs', f'{name}.wav')
                table[path] = (0, normalized)
        # read audio
        return table
