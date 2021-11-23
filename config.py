from typing import Optional


class Config:
    """Configuration for dataset construction.
    """
    def __init__(self, batch: Optional[int] = 16):
        """Initializer.
        Args:
            batch: size of the batch.
                if None is provided, single datum will be returned.
        """
        # audio config
        self.sr = 22050

        # stft
        self.fft = 1024
        self.hop = 256
        self.win = self.fft
        self.win_fn = 'hann'

        # mel-scale filter bank
        self.mel = 80
        self.fmin = 0
        self.fmax = 8000

        # for preventing log-underflow
        self.eps = 1e-5

        # sample size
        self.batch = batch
