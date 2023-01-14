import os
from typing import Dict, List, Optional, Tuple

from .reader import DataReader


class LibriTTS(DataReader):
    """LibriTTS dataset loader.
    Use other opensource settings, 16bit, sr: 24khz.
    """
    SR = 24000

    def __init__(self, data_dir: str, sr: Optional[int] = None):
        """Initializer.
        Args:
            data_dir: dataset directory.
            sr: sampling rate.
        """
        self.sr = sr or LibriTTS.SR
        self.speakers_, self.transcript = self.load_data(data_dir)

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
        return self.speakers_

    def load_data(self, data_dir: str) -> Tuple[List[str], Dict[str, Tuple[int, str]]]:
        """Load audio.
        Args:
            data_dir: dataset directory.
        Returns:
            list of speakers, transcripts.
        """
        # generate file lists
        speakers, trans = os.listdir(data_dir), {}
        for sid, speaker in enumerate(speakers):
            for chapter in os.listdir(os.path.join(data_dir, speaker)):
                path = os.path.join(data_dir, speaker, chapter)
                # read transcription
                with open(os.path.join(path, f'{speaker}_{chapter}.trans.tsv')) as f:
                    for row in f.readlines():
                        filename, _, normalized = row.replace('\n', '').split('\t')
                        fullpath = os.path.join(path, f'{filename}.wav')
                        trans[fullpath] = (sid, normalized)
        # read audio
        return speakers, trans
