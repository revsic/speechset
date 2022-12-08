from typing import Callable, List


class DataReader:
    """Interface of the data reader for efficient train-test split.
    """
    def dataset(self) -> List:
        """Return file reader.
        Returns:
            file-format datum reader, without any preprocessor for fast train-text split.
        """
        raise NotImplementedError('DataReader.rawset is not implemented')

    def preproc(self) -> Callable:
        """Return data preprocessor.
        Returns:
            preprocessor, required format,
                text: str, text.
                speech: [np.float32; T], speech signal in range (-1, 1).
        """
        raise NotImplementedError('DataReader.preproc is not implemented')

    def count_speakers(self) -> int:
        """Count the number of speakers.
        Returns:
            the number of the speakers.
        """
        raise NotImplementedError('DataReader.count_speakers is not implemented')
