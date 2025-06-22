import argparse
from pathlib import Path
import tensorflow.keras as tk
import pandas as pd
import pyfaidx
import sys
from promoterai.generator import VariantDataGenerator
from promoterai.architecture import twin_wrap
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', required=True)
    parser.add_argument('--var_file', required=True)
    parser.add_argument('--fasta_file', required=True)
    parser.add_argument('--input_length', type=int, required=True)
    args = parser.parse_args()

    try:
        model_folder = Path(args.model_folder)
        var_file = Path(args.var_file)
        model = tk.models.load_model(model_folder)
        df_var = pd.read_csv(var_file, sep='\t')
        fasta = pyfaidx.Fasta(args.fasta_file)
    except Exception as e:
        print(e)
        sys.exit(1)
    required_cols = {'chrom', 'pos', 'ref', 'alt', 'strand'}
    missing_cols = required_cols - set(df_var.columns)
    if missing_cols:
        print(f'Variant file missing column(s): {missing_cols}')
        sys.exit(1)

    gen_var = VariantDataGenerator(df_var, fasta, args.input_length, 1)
    twin_model = twin_wrap(model)
    df_var['score'] = np.tanh(twin_model.predict(gen_var).round(4))

    output_file = var_file.with_name(
        f'{var_file.stem}.{model_folder.name}{var_file.suffix}'
    )
    df_var.to_csv(output_file, sep='\t', index=False)


if __name__ == '__main__':
    main()
