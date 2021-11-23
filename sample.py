import sys
sys.path.append('..')

import speechset
sys.path.pop()


# construct data reader
lj = speechset.datasets.LJSpeech('D:\\dataset\\LJSpeech-1.1')

# construct configuration
config = speechset.Config()

# construct acoustic model
acoustic = speechset.AcousticDataset(lj, config)

# indexing
text, mel = acoustic[0]
print(text.shape, mel.shape)

# split sample
testset = acoustic.split(1000)
# back pack
text, mel, textlen, mellen = acoustic[0:3]
print(text.shape, mel.shape, textlen.shape, mellen.shape)
# iteration
text, mel = next(iter(testset))
print(text.shape, mel.shape)

# construct vocoder model
vocoder = speechset.VocoderDataset(lj, config)

# unpack
mel, audio, mellen, audiolen = vocoder[0:3]
print(mel.shape, audio.shape, mellen.shape, audiolen.shape)
