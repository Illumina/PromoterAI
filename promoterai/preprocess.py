import argparse
import pandas as pd
import pyfaidx
import pyBigWig
from promoterai.generator import SequenceDataGenerator
import os
import promoterai.tfrecords as tfrecords

parser = argparse.ArgumentParser()
parser.add_argument('--tfr_folder')
parser.add_argument('--tss_file')
parser.add_argument('--fasta_file')
parser.add_argument('--bigwig_files')
parser.add_argument('--chrom')
parser.add_argument('--input_length', type=int)
parser.add_argument('--output_length', type=int)
parser.add_argument('--chunk_size', type=int)
args = parser.parse_args()

print(args.tfr_folder)
df_tss = pd.read_csv(args.tss_file, sep='\t')
df_tss = df_tss[df_tss['chrom'] == args.chrom]
fasta = pyfaidx.Fasta(args.fasta_file)
df_bw = pd.read_csv(args.bigwig_files, sep='\t')
bws_fwd = df_bw['fwd'].apply(pyBigWig.open)
bws_rev = df_bw['rev'].apply(pyBigWig.open)
bw_xforms = df_bw['xform'].apply(eval)
gen_tss = SequenceDataGenerator(
    df_tss,
    fasta,
    bws_fwd,
    bws_rev,
    bw_xforms,
    args.input_length,
    args.output_length,
    args.chunk_size,
    shuffle=True
)

os.makedirs(args.tfr_folder, exist_ok=True)
for idx_chunk, [xs, ys] in enumerate(gen_tss):
    tfr_file = f'{args.tfr_folder}/{args.chrom}_{idx_chunk}.tfr'
    tfrecords.make_tfr_file(tfr_file, xs, ys)
