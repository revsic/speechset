import os
from typing import Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np

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
        self.speakers_, self.filelist, self.transcript = self.load_data(data_dir)

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
                sid: int, speaker id.
                text: str, text.
                speech: [np.float32; T], speech signal in range (-1, 1).
        """
        return self.preprocessor

    def speakers(self) -> List[str]:
        """Return list of speakers.
        Returns:
            list of the speakers.
        """
        return self.speakers_

    def load_data(self, data_dir: str) \
            -> Tuple[List[str], List[str], Dict[str, Tuple[int, str]]]:
        """Load audio.
        Args:
            data_dir: dataset directory.
        Returns:
            list of speakers, file paths and transcripts.
        """
        wavpath = os.path.join(data_dir, 'wav48')
        txtpath = os.path.join(data_dir, 'txt')
        # generate file lists
        speakers, paths, trans = os.listdir(wavpath), [], {}
        for sid, speaker in enumerate(speakers):
            for filename in os.listdir(os.path.join(wavpath, speaker)):
                if not filename.endswith('.wav'):
                    continue
                # for preventing exception
                if not os.path.exists(os.path.join(txtpath, speaker)):
                    continue
                # appension
                paths.append(os.path.join(wavpath, speaker, filename))
                with open(os.path.join(txtpath, speaker, filename.replace('.wav', '.txt'))) as f:
                    trans[filename.replace('.wav', '')] = (sid, f.read().strip())
        # read audio
        return speakers, paths, trans

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
        audio = self.load_audio(path, self.sr)
        # str
        path = os.path.basename(path).replace('.wav', '')
        # int, str
        sid, text = self.transcript.get(path, (-1, ''))
        # int, str, [np.float32; T]
        return sid, text, audio
