import functools
import multiprocessing as mp
import os
from typing import Any, Callable, List, Type

import numpy as np
from tqdm import tqdm

from .. import Config
from ..datasets import DataReader
from ..speeches.speechset import SpeechSet


class DumpReader(DataReader):
    """Dumped dataset reader.
    """
    def __init__(self, data_dir: str):
        """Initializer.
        Args:
            data_dir: dumped datasets.
        """
        self.filelists = [
            os.path.join(data_dir, path)
            for path in os.listdir(data_dir)
            if path.endswith('.npy')]

    def dataset(self) -> List[str]:
        """Return file reader.
        Returns:
            file-format datum read.er
        """
        return self.filelists
    
    def preproc(self) -> Callable:
        """Return data preprocessor.
        Returns;
            preprocessor, load npy and return normalized.
        """
        def preprocessor(path: str) -> Any:
            """Load umped datum.
            Args:
                path: path to the npy file.
            Returns:
                loaded.
            """
            return np.load(path, allow_pickle=True)

        return preprocessor


class DumpDataset(SpeechSet):
    """Dumped dataset.
    """
    def __init__(self, settype: Type[SpeechSet], data_dir: str):
        """Initializer.
        WARNING: given speechset type should support `Config` independent `collate`.
        Args:
            settype: Type of the speechset.
            data_dir: path to the dumped dataset.
        """
        # use dummy config
        self.speechset = settype(DumpReader(data_dir), Config())
        # cache
        self.dataset, self.preproc = self.speechset.dataset, self.speechset.preproc

    def normalize(self, *args) -> List[Any]:
        """Identity map.
        """
        return args

    def collate(self, bunch: List[Any]) -> Any:
        """Collator.
        """
        return self.speechset.collate(bunch)


def dumper(speechset: SpeechSet, outdir: str, i: int) -> int:
    """Dump the datum of specified index.
    Args:
        speechset: target dataset.
        outdir: output path.
        i: index.
    Returns:
        dumped index, same with argument `i`.
    """
    np.save(os.path.join(outdir, f'{i}.npy'), speechset[i])
    return i


def mp_dump(speechset: SpeechSet,
            outdir: str,
            num_proc: int,
            chunksize: int = 1) -> int:
    """Dump dataset.
    Args:
        speechset: target dataset.
        outdir: path to the output directory.
        num_proc: the number of the process for multiprocessing.
        chunksize: size of the imap_ordered chunk.
    Returns:
        the number of the written data.
    """
    os.makedirs(outdir, exist_ok=True)
    partial = functools.partial(dumper, speechset, outdir)

    length = len(speechset)
    with mp.Pool(num_proc) as pool:
        worker = pool.imap_unordered(partial, range(length), chunksize=chunksize)
        for _ in tqdm(worker, total=length):
            pass

    return length
