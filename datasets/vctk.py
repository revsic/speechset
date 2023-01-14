import os
from typing import Dict, List, Optional, Tuple

from .reader import DataReader


class VCTK(DataReader):
    """VCTK dataset loader.
    Use other opensource settings, 16bit, sr: 48khz.
    """
    SR = 48000

    def __init__(self, data_dir: str, sr: Optional[int] = None):
        """Initializer.
        Args:
            data_dir: dataset directory.
            sr: sampling rate.
        """
        self.sr = sr or VCTK.SR
        self.speakers_, self.transcript = self.load_data(data_dir)

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
        return self.speakers_

    def load_data(self, data_dir: str) \
            -> Tuple[List[str], Dict[str, Tuple[int, str]]]:
        """Load audio.
        Args:
            data_dir: dataset directory.
        Returns:
            list of speakers, transcripts.
        """
        wavpath = os.path.join(data_dir, 'wav48')
        txtpath = os.path.join(data_dir, 'txt')
        # generate file lists
        speakers, trans = os.listdir(wavpath), {}
        for sid, speaker in enumerate(speakers):
            for filename in os.listdir(os.path.join(wavpath, speaker)):
                if not filename.endswith('.wav'):
                    continue
                # for preventing exception
                if not os.path.exists(os.path.join(txtpath, speaker)):
                    continue
                # appension
                path = os.path.join(wavpath, speaker, filename)
                with open(os.path.join(txtpath, speaker, filename.replace('.wav', '.txt'))) as f:
                    trans[path] = (sid, f.read().strip())
        # read audio
        return speakers, trans
