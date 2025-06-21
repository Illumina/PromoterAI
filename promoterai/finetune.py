import argparse
import pandas as pd
import pyfaidx
from promoterai.generator import VariantDataGenerator
import tensorflow as tf
import tensorflow.keras as tk
from promoterai.architecture import twin_wrap
import os
from promoterai.utils import CustomLearningRateScheduler, CustomModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--model_folder')
parser.add_argument('--var_file')
parser.add_argument('--fasta_file')
parser.add_argument('--input_length', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--learning_rate', type=float, default=5e-4)
parser.add_argument('--weight_decay', type=float, default=5e-6)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

twin_model_folder = args.model_folder + '_finetune'
print(args.model_folder, twin_model_folder)
df_var = pd.read_csv(args.var_file, sep='\t')
df_var = df_var[
    (df_var['in_cds'] == 0) & (df_var['spliceai'] < 0.05)
]
df_outlier = df_var[(df_var['p_under'] < 0.01) | (df_var['p_over'] < 0.01)]
df_var = df_var[df_var['gene'].isin(df_outlier['gene'])]
df_train = df_var[df_var['chrom'].isin([f'chr{i}' for i in range(1, 21, 2)])]
df_valid = df_var[df_var['chrom'].isin([f'chr{i}' for i in range(21, 23)])]
fasta = pyfaidx.Fasta(args.fasta_file)
gen_train = VariantDataGenerator(
    df_train,
    fasta,
    args.input_length,
    args.batch_size,
    output='z',
    shuffle=True
)
gen_valid = VariantDataGenerator(
    df_valid, fasta, args.input_length, args.batch_size, output='z'
)

strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    model = tk.models.load_model(args.model_folder)
    twin_model = twin_wrap(model)
    twin_model.compile(optimizer=tk.optimizers.AdamW(clipnorm=1), loss='mse')

os.makedirs(twin_model_folder)
scheduler = CustomLearningRateScheduler(
    args.learning_rate, args.weight_decay, args.epochs
)
checkpoint = CustomModelCheckpoint(model, twin_model_folder)
logger = tk.callbacks.CSVLogger(f'{twin_model_folder}/logs.csv', append=True)
twin_model.summary()
twin_model.fit(
    gen_train,
    validation_data=gen_valid,
    epochs=args.epochs,
    steps_per_epoch=len(gen_train) // 5,
    callbacks=[scheduler, checkpoint, logger],
    verbose=2
)
