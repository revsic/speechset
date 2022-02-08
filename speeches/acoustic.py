from typing import List, Optional, Tuple

import numpy as np

from .speechset import SpeechSet
from ..config import Config
from ..datasets import DataReader
from ..utils import MelSTFT, TextNormalizer


class AcousticDataset(SpeechSet):
    """Dataset for text to acoustic features.
    """
    VOCABS = len(TextNormalizer.GRAPHEMES) + 1

    def __init__(self,
                 rawset: DataReader,
                 config: Config,
                 report_level: Optional[int] = None):
        """Initializer.
        Args:
            rawset: file-format datum reader.
            config: configuration.
            report_level: text normalizing error report level.
        """
        # cache dataset and preprocessor
        super().__init__(rawset)
        self.config = config
        self.melstft = MelSTFT(config)
        self.textnorm = TextNormalizer(report_level)

    def normalize(self, text: str, speech: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Normalize datum.
        Args:
            text: transcription.
            speech: [np.float32; [T]], speech in range (-1, 1).
        Returns:
            normalized datum.
                labels: [np.long; [S]], labeled text sequence.
                mel: [np.float32; [T // hop, mel]], mel spectrogram.
        """
        # [S]
        labels = np.array(self.textnorm.labeling(text), dtype=np.long)
        # [T // hop, mel]
        mel = self.melstft(speech)
        return labels, mel

    def collate(self, bunch: List[Tuple[np.ndarray, np.ndarray]]) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collate bunch of datum to the batch data.
        Args:
            bunch: B x [...] list of normalized inputs.
                labels: [np.long; [Si]], labled text sequence.
                mel: [np.float32; [Ti, mel]], mel spectrogram.
        Returns:
            batch data.
                text: [np.long; [B, S]], labeled text sequence.
                mel: [np.float32; [B, T, mel]], mel spectrogram.
                textlen: [np.long; [B]], text lengths.
                mellen: [np.long; [B]], spectrogram lengths.
        """
        # [B], [B]
        textlen, mellen = np.array(
            [[len(labels), len(spec)] for labels, spec in bunch], dtype=np.long).T
        # [B, S]
        text = np.stack(
            [np.pad(labels, [0, textlen.max() - len(labels)]) for labels, _ in bunch])
        # [B, T, mel]
        mel = np.stack(
            [np.pad(spec, [[0, mellen.max() - len(spec)], [0, 0]]) for _, spec in bunch])
        return mel, text, textlen, mellen
