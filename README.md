# speechset

Tensorflow implementation of Speech dataset pipeline

## Requirements

Tested in python 3.8.5 windows 10 miniconda environment, [requirements.txt](./requirements.txt)

## Sample

Sample script is provided as [sample.py](./sample.py)

## TFRecord Suuport

Dump the tfrecord.
```
python dump.py \
    --reader ljspeech \
    --data-dir D:\dataset\LJSpeech-1.1\tfrecord \
    --from-raw \
    --target acoustic \
    --path D:\dataset\LJSpeech-1.1\tfrecord\acoustic
```

Load TFRecord
```python
record = speechset.TFRecordDataset(
    config, 'D:\\dataset\\ljspeech.tfrecord', 'acoustic')
trainset, testset = record.dataset(13000)
# unpack
text, mel, textlen, mellen = next(iter(testset))
```
