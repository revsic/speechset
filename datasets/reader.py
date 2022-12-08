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
                (optional) sid: int, speaker id.
                text: str, text.
                speech: [np.float32; T], speech signal in range (-1, 1).
        """
        raise NotImplementedError('DataReader.preproc is not implemented')

    def speakers(self) -> List[str]:
        """Return list of speakers.
        Returns:
            list of the speakers.
        """
        raise NotImplementedError('DataReader.speakers is not implemented')
