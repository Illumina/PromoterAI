import argparse
import pandas as pd
import sys
import pyfaidx
try:
    from promoterai.generator import VariantDataGenerator
except ImportError:
    from generator import VariantDataGenerator
import tensorflow.keras as tk
try:
    from promoterai.architecture import twin_wrap
except ImportError:
    from architecture import twin_wrap
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', required=True)
    parser.add_argument('--var_file', required=True)
    parser.add_argument('--fasta_file', required=True)
    parser.add_argument('--input_length', type=int, required=True)
    args = parser.parse_args()

    try:
        df_var = pd.read_csv(args.var_file, sep='\t')
    except OSError as e:
        print(e)
        sys.exit(1)
    required_cols = {'chrom', 'pos', 'ref', 'alt', 'strand'}
    missing_cols = required_cols - set(df_var.columns)
    if missing_cols:
        print(f'Variant file missing {missing_cols} columns')
        sys.exit(1)
    try:
        fasta = pyfaidx.Fasta(args.fasta_file)
    except OSError as e:
        print(e)
        sys.exit(1)
    gen_var = VariantDataGenerator(df_var, fasta, args.input_length, 1)

    try:
        model = tk.models.load_model(args.model_folder)
    except OSError as e:
        print(e)
        sys.exit(1)
    twin_model = twin_wrap(model)

    df_var['score'] = np.tanh(twin_model.predict(gen_var).round(4))
    model_name = args.model_folder.split('/')[-1]
    df_var.to_csv(f'{args.var_file}.{model_name}', sep='\t', index=False)


if __name__ == '__main__':
    main()
