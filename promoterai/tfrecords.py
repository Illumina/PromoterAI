import tensorflow as tf
from functools import partial


AUTOTUNE = tf.data.AUTOTUNE


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


_feature_fns = {
    'x': {
        'encode':
            lambda x, y: _bytes_feature([x.astype('float32').tobytes()]),
        'decode':
            lambda example: tf.reshape(
                tf.io.decode_raw(example['x'], 'float32'), example['x_shape']
            ),
        'description': tf.io.FixedLenFeature([], 'string')
    },
    'y': {
        'encode':
            lambda x, y: _bytes_feature([y.astype('float32').tobytes()]),
        'decode':
            lambda example: tf.reshape(
                tf.io.decode_raw(example['y'], 'float32'), example['y_shape']
            ),
        'description': tf.io.FixedLenFeature([], 'string')
    },
    'x_shape': {
        'encode': lambda x, y: _int64_feature(x.shape),
        'decode': lambda example: example['x_shape'],
        'description': tf.io.FixedLenFeature([2], 'int64')
    },
    'y_shape': {
        'encode': lambda x, y: _int64_feature(y.shape),
        'decode': lambda example: example['y_shape'],
        'description': tf.io.FixedLenFeature([2], 'int64')
    }
}


def _serialize_example(x, y):
    feature = {
        key: value['encode'](x, y) for key, value in _feature_fns.items()
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    example = example.SerializeToString()
    return example


def _deserialize_example(example):
    feature_description = {
        key: value['description'] for key, value in _feature_fns.items()
    }
    example = tf.io.parse_example(example, feature_description)
    feature = {
        key: value['decode'](example) for key, value in _feature_fns.items()
    }
    x, y = feature['x'], feature['y']
    return x, y


def make_tfr_file(tfr_file, xs, ys):
    with tf.io.TFRecordWriter(tfr_file, options='ZLIB') as writer:
        for x, y in zip(xs, ys):
            writer.write(_serialize_example(x, y))


def _prepare_sample(x, y, input_length, output_length, sample_weight, augment):
    input_crop = tf.cast(tf.shape(x)[0], 'int64') - input_length
    output_crop = tf.cast(tf.shape(y)[0], 'int64') - output_length
    shift_range = tf.cast(tf.math.minimum(input_crop, output_crop), 'float32')
    shift = tf.cast(
        tf.random.truncated_normal([], stddev=shift_range // 4 * augment),
        'int64'
    )
    strand = -1 + 2 * tf.cast(tf.random.uniform([]) > 0.25 * augment, 'int64')
    x = x[shift + input_crop // 2:shift - input_crop // 2][::strand, ::strand]
    y = y[shift + output_crop // 2:shift - output_crop // 2][::strand]

    y = tuple(y if sw else [[0.]] for sw in sample_weight)
    sample_weight = tuple(sw * tf.reduce_max(x) for sw in sample_weight)
    return x, y, sample_weight


def make_dataset(
        tfr_files,
        input_length,
        output_length,
        sample_weight,
        batch_size,
        augment=False
):
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.shuffle(len(dataset))
    dataset = dataset.interleave(
        partial(tf.data.TFRecordDataset, compression_type='ZLIB'),
        num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.map(_deserialize_example, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(
        partial(
            _prepare_sample,
            input_length=input_length,
            output_length=output_length,
            sample_weight=sample_weight,
            augment=augment
        ),
        num_parallel_calls=AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset
