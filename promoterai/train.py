import argparse
import promoterai.tfrecords as tfrecords
from glob import glob
import tensorflow as tf
from promoterai.architecture import promoterAI
import tensorflow.keras as tk
import os
from promoterai.utils import CustomLearningRateScheduler, CustomModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--model_folder')
parser.add_argument('--tfr_human_folder')
parser.add_argument('--tfr_nonhuman_folders', nargs='+', default=[])
parser.add_argument('--input_length', type=int)
parser.add_argument('--output_length', type=int)
parser.add_argument('--num_blocks', type=int)
parser.add_argument('--model_dim', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=5e-6)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

print(args.model_folder)
tfr_folders = [args.tfr_human_folder] + args.tfr_nonhuman_folders
datasets_train = [[] for _ in tfr_folders]
for j, tfr_folder in enumerate(tfr_folders):
    datasets_train[j] = tfrecords.make_dataset(
        sum([glob(f'{tfr_folder}/chr{i}_*') for i in range(1, 21)], start=[])
        if j == 0 else glob(f'{tfr_folder}/chr*'),
        args.input_length,
        args.output_length,
        tuple(k == j for k in range(len(tfr_folders))),
        args.batch_size,
        augment=True
    )
dataset_valid = tfrecords.make_dataset(
    sum([glob(f'{tfr_folders[0]}/chr{i}_*') for i in range(21, 23)], start=[]),
    args.input_length,
    args.output_length,
    tuple(k == 0 for k in range(len(tfr_folders))),
    args.batch_size
)
output_dims = [
    next(iter(ds))[1][j].shape[-1] for j, ds in enumerate(datasets_train)
]
output_size = [
    float(ds.reduce(0, lambda x, _: x + 1).numpy()) for ds in datasets_train
]

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = promoterAI(
        args.num_blocks,
        args.model_dim,
        output_dims,
        output_crop=args.input_length - args.output_length
    )
    model.compile(optimizer=tk.optimizers.AdamW(clipnorm=1e-4), loss='mse')

os.makedirs(args.model_folder)
scheduler = CustomLearningRateScheduler(
    args.learning_rate, args.weight_decay, args.epochs
)
checkpoint = CustomModelCheckpoint(model, args.model_folder)
logger = tk.callbacks.CSVLogger(f'{args.model_folder}/logs.csv', append=True)
model.summary()
model.fit(
    tf.data.Dataset.sample_from_datasets(
        [ds.repeat() for ds in datasets_train], weights=output_size
    ),
    validation_data=dataset_valid,
    epochs=args.epochs,
    steps_per_epoch=int(sum(output_size) / 10),
    callbacks=[scheduler, checkpoint, logger],
    verbose=2
)
