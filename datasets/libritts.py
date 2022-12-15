import os
from typing import Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np

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
        """List of speakers.
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
        # generate file lists
        paths, trans = [], {}
        speakers = os.listdir(data_dir)
        for sid, speaker in enumerate(speakers):
            for chapter in os.listdir(os.path.join(data_dir, speaker)):
                path = os.path.join(data_dir, speaker, chapter)
                # read transcription
                with open(os.path.join(path, f'{speaker}_{chapter}.trans.tsv')) as f:
                    for row in f.readlines():
                        filename, _, normalized = row.replace('\n', '').split('\t')
                        trans[filename] = (sid, normalized)
                # wav files
                paths.extend([
                    os.path.join(path, filename)
                    for filename in os.listdir(path) if filename.endswith('.wav')])
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
        audio, _ = librosa.load(path, sr=self.sr)
        # str
        path = os.path.basename(path).replace('.wav', '')
        # int, str
        sid, text = self.transcript.get(path, (-1, ''))
        # int, str, [np.float32; T]
        return sid, text, audio.astype(np.float32)
