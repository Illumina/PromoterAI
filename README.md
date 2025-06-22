# PromoterAI

This repository contains the source code for PromoterAI, a deep learning model for predicting the impact of promoter variants on gene expression, as described in [Jaganathan, Ersaro, Novakovsky et al., *Science* (2025)](https://www.science.org/doi/10.1126/science.ads7373).

PromoterAI precomputed scores for all human promoter single nucleotide variants are freely available for academic and non-commercial research use. Please complete the [license agreement](https://illumina2.na1.adobesign.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhAuRnD5FtTNwyNo-5X6njTJqQOOMu3V_0nU0MjxSi_9PLCrquWaKSRrT3e1RhHkr7w*); the download link will be shared via email shortly after submission. Scores range from –1 to 1, with negative values indicating under-expression and positive values indicating over-expression. Recommended thresholds are ±0.1, ±0.2, and ±0.5.

## Installation

The simplest way to install PromoterAI for variant effect prediction is through:
```sh
pip install promoterai
```
For model training or to work directly with the source code, install PromoterAI by cloning the repository:
```sh
git clone https://github.com/Illumina/PromoterAI
cd PromoterAI
pip install -e .
```
PromoterAI supports both CPU and GPU execution, and has been tested on H100 (TensorFlow 2.15, CUDA 12.2, cuDNN 8.9.7) and A100 (TensorFlow 2.13, CUDA 11.4, cuDNN 8.6.0) GPUs. Functionality on other GPUs is expected but not officially tested.

## Variant effect prediction

Organize the variants into a `tsv` file with the following columns: `chrom`, `pos`, `ref`, `alt`, `strand`. If strand cannot be specified, create separate rows for each strand and aggregate predictions. Indels must be left-normalized and without special characters.
```tsv
chrom	pos	ref	alt	strand
chr16	84145214	G	T	1
chr16	84145333	G	C	1
chr2	55232249	T	G	-1
chr2	55232374	C	T	-1
chr6	108295024	C	CGG	1
chr6	108295024	CT	C	1
```
Download the appropriate reference genome `fa` file, then run:
```sh
promoterai \
    --model_folder path/to/model \
    --var_file path/to/variant_tsv \
    --fasta_file path/to/genome_fa \
    --input_length 20480
```
Scores will be added as a new column labeled `score`, and written to a file created by appending the model folder name to the variant file name.

## Model training and fine-tuning

Create a `tsv` file listing the genomic positions of interest (e.g., `data/annotation/tss_hg38.tsv`, `data/annotation/tss_mm10.tsv`), with the following required columns: `chrom`, `pos`, `strand`.
```tsv
chrom	pos	strand
chr1	11868	1
chr1	12009	1
chr1	29569	-1
chr1	17435	-1
```
Download the appropriate reference genome `fa` file and regulatory profile `bigwig` files. Organize the `bigwig` file paths and their corresponding transformations into a `tsv` file (e.g., `data/bigwig/hg38.tsv`, `data/bigwig/mm10.tsv`), where each row represents a prediction target, with the following required columns:  
- `fwd`: path to the forward-strand `bigwig` file  
- `rev`: path to the reverse-strand `bigwig` file  
- `xform`: transformation applied to the prediction target  
```tsv
fwd	rev	xform
path/to/ENCFF245ZZX.bigWig	path/to/ENCFF245ZZX.bigWig	lambda x: np.arcsinh(np.nan_to_num(x))
path/to/ENCFF279QDX.bigWig	path/to/ENCFF279QDX.bigWig	lambda x: np.arcsinh(np.nan_to_num(x))
path/to/ENCFF480GFU.bigWig	path/to/ENCFF480GFU.bigWig	lambda x: np.arcsinh(np.nan_to_num(x))
path/to/ENCFF815ONV.bigWig	path/to/ENCFF815ONV.bigWig	lambda x: np.arcsinh(np.nan_to_num(x))
```
Generate TFRecord files by running the command below, which can be parallelized across chromosomes for speed:
```sh
for chrom in $(cut -f1 path/to/position_tsv | sort -u | grep -v chrom)
do
    python -m promoterai.preprocess \
        --tfr_folder path/to/output_tfrecord \
        --tss_file path/to/position_tsv \
        --fasta_file path/to/genome_fa \
        --bigwig_files path/to/profile_tsv \
        --chrom ${chrom} \
        --input_length 32768 \
        --output_length 16384 \
        --chunk_size 256
done
```
For multi-species training, repeat the steps above for each species, writing TFRecord files to separate folders. Train a model on the generated TFRecord files by running:
```sh
python -m promoterai.train \
    --model_folder path/to/trained_model \
    --tfr_human_folder path/to/human_tfrecord \
    --input_length 20480 \
    --output_length 4096 \
    --num_blocks 24 \
    --model_dim 1024 \
    --batch_size 32 \
    --tfr_nonhuman_folders [path/to/mouse_tfrecord ...]  # optional list
```
Fine-tune the trained model using the variant file `data/annotation/finetune_gtex.tsv` by running:
```sh
python -m promoterai.finetune \
    --model_folder path/to/trained_model \
    --var_file path/to/finetune_gtex_tsv \
    --fasta_file path/to/genome_fa \
    --input_length 20480 \
    --batch_size 8
```
The fine-tuned model will be saved in a folder created by appending `_finetune` to the trained model folder name.

## Contact

- Kishore Jaganathan: [kjaganathan@illumina.com](mailto:kjaganathan@illumina.com)  
- Gherman Novakovsky: [gnovakovsky@illumina.com](mailto:gnovakovsky@illumina.com)  
- Kyle Farh: [kfarh@illumina.com](mailto:kfarh@illumina.com)
