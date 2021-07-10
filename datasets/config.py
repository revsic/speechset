from typing import Callable

import tensorflow as tf


class Config:
    """Configuration for dataset construction.
    """
    def __init__(self):
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
        self.batch = 16

    def window_fn(self) -> Callable:
        """Return window generator.
        Returns:
            window function of tf.signal
                , which corresponds to self.win_fn.
        """
        mapper = {
            'hann': tf.signal.hann_window,
            'hamming': tf.signal.hamming_window
        }
        if self.win_fn in mapper:
            return mapper[self.win_fn]
        
        raise ValueError('invalid window function: ' + self.win_fn)
