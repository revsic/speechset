import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .reader import DataReader


class LibriSpeech(DataReader):
    """LibriSpeech dataset loader.
    Use other opensource settings, 16bit, sr: 16khz.
    """
    SR = 16000

    def __init__(self, data_dir: str, sr: Optional[int] = None):
        """Initializer.
        Args:
            data_dir: dataset directory.
            sr: sampling rate.
        """
        self.sr = sr or LibriSpeech.SR
        self.speakers_, self.transcript = self.load_data(data_dir)

    def dataset(self) -> List[str]:
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
        return self.speakers_

    def load_data(self, data_dir: str) -> Tuple[List[str], Dict[str, Tuple[int, str]]]:
        """Load audio.
        Args:
            data_dir: dataset directory.
        Returns:
            loaded data, speaker list, transcripts.
        """
        # generate file lists
        speakers, trans = os.listdir(data_dir), {}
        for sid, speaker in enumerate(speakers):
            for chapter in os.listdir(os.path.join(data_dir, speaker)):
                path = os.path.join(data_dir, speaker, chapter)
                # read transcription
                with open(os.path.join(path, f'{speaker}-{chapter}.trans.txt')) as f:
                    for row in f.readlines():
                        filename, *text = row.replace('\n', '').split(' ')
                        # re-aggregation
                        text = ' '.join(text).strip()
                        fullpath = os.path.join(path, f'{filename}.flac')
                        trans[fullpath] = (sid, text)
        # read audio
        return speakers, trans
