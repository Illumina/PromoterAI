import argparse
import pandas as pd
import pyfaidx
from generator import VariantDataGenerator
import tensorflow.keras as tk
from architecture import twin_wrap
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder')
    parser.add_argument('--var_file')
    parser.add_argument('--fasta_file')
    parser.add_argument('--input_length', type=int)
    args = parser.parse_args()

    df_var = pd.read_csv(args.var_file, sep='\t')
    fasta = pyfaidx.Fasta(args.fasta_file)
    gen_var = VariantDataGenerator(df_var, fasta, args.input_length, 1)

    model = tk.models.load_model(args.model_folder)
    twin_model = twin_wrap(model)
    df_var['score'] = np.tanh(twin_model.predict(gen_var).round(4))

    model_name = args.model_folder.split('/')[-1]
    df_var.to_csv(f'{args.var_file}.{model_name}', sep='\t', index=False)


if __name__ == '__main__':
    main()
