import sys
sys.path.append('..')

import speechset
sys.path.pop()


# construct data reader
lj = speechset.datasets.LJSpeech()

# construct configuration
config = speechset.Config()

# construct acoustic model
acoustic = speechset.AcousticDataset(lj, config)

# unpack
text, mel, textlen, mellen = acoustic[0]
print(text.shape, mel.shape, textlen.shape, mellen.shape)

# split sample
testset = acoustic.split(1000)
# unpack
text, mel, textlen, mellen = acoustic.collate(acoustic[0:3])
print(text.shape, mel.shape, textlen.shape, mellen.shape)

text, mel, textlen, mellen = next(iter(testset))
print(text.shape, mel.shape, textlen.shape, mellen.shape)

# construct vocoder model
vocoder = speechset.VocoderDataset(lj, config)

# unpack
mel, audio, mellen, audiolen = next(iter(vocoder))
print(mel.shape, audio.shape, mellen.shape, audiolen.shape)
