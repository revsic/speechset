from typing import List, Tuple

import numpy as np

from .speechset import SpeechSet
from ..config import Config
from ..datasets.reader import DataReader
from ..utils.melstft import MelSTFT


class VocoderDataset(SpeechSet):
    """Dataset for acoustic features to audio signal.
    """
    def __init__(self, rawset: DataReader, config: Config):
        """Initializer.
        Args:
            rawset: file-format datum reader.
            config: configuration.
        """
        super().__init__(rawset)
        self.config = config
        self.melstft = MelSTFT(config)

    def normalize(self, _: str, speech: np.Tensor) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Normalize datum.
        Args:
            _: str, placeholder, transcription.
            speech: [np.float32; [T]], speech in range (-1, 1.)
        Returns:
            normalized datum.
                mel: [np.float32; [T // hop + 1, mel]], mel spectrogram.
                speech: [np.float32; [T]], speech signal.
        """
        # [T // hop + 1, mel]
        return self.melstft(speech), speech

    def collate(self, bunch: List[Tuple[np.ndarray, np.ndarray]]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collate bunch of datum to the batch data.
        Args:
            bunch: B x [...] list of normalized inputs.
                mel: [np.float32; [T // hop + 1, mel]], mel spectrogram.
                speech: [np.float32; [T]], speech signal.
        Returns:
            batch data.
                mel: [np.float32; [B, T // hop + 1, mel]], mel spectrogram.
                speech: [np.float32; [B, T]], speech signal.
                mellen: [np.long; [B]], spectrogram lengths.
                speechlen: [np.long; [B]], signal lengths.
        """
        # [B], [B]
        mellen, speechlen = np.array(
            [[len(spec), len(signal)] for spec, signal in bunch], dtype=np.long).T
        # [B, T, mel]
        mel = np.stack(
            [np.pad(spec, [[0, len(spec) - mellen.max()], [0, 0]]) for spec, _ in bunch])
        # [B, S]
        speech = np.stack(
            [np.pad(signal, [0, len(signal) - speechlen.max()]) for _, signal in bunch])
        return mel, speech, mellen, speechlen
