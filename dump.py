import sys
sys.path.append('..')

import argparse
import json

import tensorflow as tf

import speechset


parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None)
parser.add_argument('--reader', default='ljspeech')
parser.add_argument('--data-dir', default=None)
parser.add_argument('--download', default=False, action='store_true')
parser.add_argument('--from-raw', default=False, action='store_true')
parser.add_argument('--target', default='acoustic')
parser.add_argument('--path', default=None)
args = parser.parse_args()

# read config
config = speechset.Config()
if args.config is not None:
    print('[*] load config: ' + args.config)
    with open(args.config) as f:
        for k, v in json.load(f).items():
            if hasattr(config, k):
                setattr(config, k, v)
config.batch = None

# dump config
assert args.path is not None, 'args.path should not be None'
with open(f'{args.path}.json', 'w') as f:
    json.dump(vars(config), f)

# construct data reader
READER_SET = {
    'ljspeech': speechset.datasets.LJSpeech}
assert args.reader in READER_SET, 'args.reader should be one of the READER_SET'
reader = READER_SET[args.reader](args.data_dir, args.download, not args.from_raw)

# construct dataset
TARGET_SET = {'acoustic': speechset.AcousticDataset, 'vocoder': speechset.VocoderDataset}
assert args.target in TARGET_SET, 'args.target should be one of the [acoustic, vocoder].'
model = TARGET_SET[args.target](reader, config)


# generate dataset
def serializer(dataset):
    total = tf.cast(tf.data.experimental.cardinality(dataset), dtype=tf.float32)
    idx = tf.Variable(tf.convert_to_tensor(0.))
    def inner(*tensors):
        dep = [
            idx.assign_add(1.),
            tf.print('\r', idx / total, end='')]
        with tf.control_dependencies(dep):
            return tf.io.serialize_tensor([tf.io.serialize_tensor(x) for x in tensors])
    return dataset.map(inner)

dataset, _ = model.dataset()
writer = tf.data.experimental.TFRecordWriter(f'{args.path}.tfrecord')
writer.write(serializer(dataset))

print('[*] success')

"""
python dump.py --reader ljspeech --data-dir D:\dataset\LJSpeech-1.1 --from-raw --target acoustic --path D:\dataset\LJSpeech-1.1\tfrecord\acoustic
"""
