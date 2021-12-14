from typing import List, Optional


class TextNormalizer:
    """Normalize text in to fixed graphme set.
    WARNING: It does not accept digits, please use normalized text in LJ Speech.
    """
    GRAPHEMES = 'abcdefghijklmnopqrstuvwxyz !?,.'
    REPLACER = {
        '"\'()-:;[]’“”': '',                # punct
        'àâ': 'a', 'èéê': 'e', 'ü': 'u'     # special character in ljspeech.
    }

    REPORT_ERROR = 0
    REPORT_LOG = 1

    def __init__(self, report_level: Optional[int] = None):
        """Initializer.
        Args:
            report_level: whether write log or raise error when OOD,
                error default.
        """
        replacer = {}
        for rep, out in TextNormalizer.REPLACER.items():
            for r in rep:
                replacer[r] = out
        self.replacer = replacer
        # set default
        self.report_level = report_level or TextNormalizer.REPORT_ERROR

    def grapheme_fn(self, grapheme: str) -> str:
        """Map grapheme into fixed set `TextNormalizer.GRAPHEMES`.
        Args:
            grapheme: single grapheme.
        Returns:
            normalized form.
        """
        if grapheme in self.replacer:
            grapheme = self.replacer[grapheme]
        if grapheme not in TextNormalizer.GRAPHEMES:
            msg = f'invalid grapheme: {grapheme}'
            if self.report_level == TextNormalizer.REPORT_ERROR:
                raise RuntimeError(msg)
            print(f'[*] speechset.utils.normalizer.TextNormalizer: {msg}')
            # use blank
            grapheme = ''
        return grapheme

    def normalize(self, text: str) -> str:
        """Normalize text.
        Args:
            text: input text.
        Returns:
            normalized.
        """
        return ''.join(self.grapheme_fn(t.lower()) for t in text)

    def labeling(self, text: str) -> List[int]:
        """Normalize text and make to integer label.
        Padding token for 0, TextNormalizer.GRAPHEMES for afters.

        Args:
            text: input text.
        Returns:
            integer labels.
        """
        return [
            TextNormalizer.GRAPHEMES.index(t) + 1
            for t in self.normalize(text)]

    def recover(self, labels: List[int]) -> str:
        """Convert label to normalized text.
        Ars:
            labels: integer label.
        Returns:
            converted text.
        """
        return ''.join(
            TextNormalizer.GRAPHEMES[i - 1] 
            for i in labels if i != 0)
