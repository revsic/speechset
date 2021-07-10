import sys
sys.path.append('..')

import speechset


# construct data reader
lj = speechset.datasets.LJSpeech()

# construct configuration
config = speechset.Config()

# construct acoustic model
acoustic = speechset.AcousticDataset(lj, config)

# generate tf.data.Dataset
dataset, _ = acoustic.dataset()
# unpack
text, mel, textlen, mellen = next(iter(dataset))
print(text.shape, mel.shape, textlen.shape, mellen.shape)

# split sample
trainset, testset = acoustic.dataset(1000)
# unpack
text, mel, textlen, mellen = next(iter(trainset))
print(text.shape, mel.shape, textlen.shape, mellen.shape)

text, mel, textlen, mellen = next(iter(testset))
print(text.shape, mel.shape, textlen.shape, mellen.shape)

# construct vocoder model
vocoder = speechset.VocoderDataset(lj, config)

# generate tf.data.Dataset
dataset, _ = vocoder.dataset()
# unpack
mel, audio, mellen, audiolen = next(iter(dataset))
print(mel.shape, audio.shape, mellen.shape, audiolen.shape)
