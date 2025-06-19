# PromoterAI

This repository contains the source code for PromoterAI, a deep learning model for predicting the impact of promoter variants on gene expression, as described in [Jaganathan, Ersaro, Novakovsky et al., *Science* (2025)](https://www.science.org/doi/10.1126/science.ads7373).

PromoterAI precomputed scores for all human promoter single nucleotide variants are freely available for academic and non-commercial research use. Please complete the [license agreement](https://illumina2.na1.adobesign.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhAuRnD5FtTNwyNo-5X6njTJqQOOMu3V_0nU0MjxSi_9PLCrquWaKSRrT3e1RhHkr7w*); the download link will be shared via email shortly after submission. Scores range from –1 to 1, with negative values indicating under-expression and positive values indicating over-expression. Recommended thresholds are ±0.1, ±0.2, and ±0.5.

## Installation

The simplest way to install PromoterAI for variant scoring is via:
```bash
pip install promoterai
```
For model training or to work directly with the source code, install PromoterAI by cloning the repository:
```bash
git clone https://github.com/Illumina/PromoterAI
cd PromoterAI
python setup.py install
```
PromoterAI supports both CPU and GPU execution, and has been tested on H100 (TensorFlow 2.15, CUDA 12.2, cuDNN 8.9.7) and A100 (TensorFlow 2.13, CUDA 11.4, cuDNN 8.6.0) GPUs. A quick check to confirm proper setup (especially when using a different GPU or environment) is to run:
```bash
python -c "import tensorflow"
```

## Variant scoring

To score variants, organize them into a `.tsv` file with the following columns: `chrom`, `pos`, `ref`, `alt`, `strand`. If strand cannot be specified, create separate rows for each strand and aggregate predictions. Indels must be left-normalized.
```tsv
chrom   pos     ref     alt     strand
chr16   84145214        G       T       1
chr16   84145333        G       C       1
chr2    55232249        T       G       -1
chr2    55232374        C       T       -1
```
Download the appropriate reference genome `.fa` file, and run the following command:
```bash
promoterai \
    --model_folder path/to/model_dir \
    --var_file path/to/variant_tsv \
    --fasta_file path/to/genome_fa \
    --input_length 20480
```
Scores will be added as a new column labeled `score`, with the output file named by appending the model folder’s basename to the variant file name.

## Model training

To begin, download the appropriate reference genome `.fa` file and regulatory profile `.bigWig` files. Organize the `.bigWig` file paths and their corresponding transformations into a `.tsv` file, where each row represents a prediction target, with the following columns:  
- `fwd`: path to the forward-strand `.bigWig` file  
- `rev`: path to the reverse-strand `.bigWig` file  
- `xform`: transformation applied to the prediction target  
```tsv
fwd	rev	xform
data/bigwig/ENCFF245ZZX.bigWig	data/bigwig/ENCFF245ZZX.bigWig	lambda x: np.arcsinh(np.nan_to_num(x))
data/bigwig/ENCFF279QDX.bigWig	data/bigwig/ENCFF279QDX.bigWig	lambda x: np.arcsinh(np.nan_to_num(x))
data/bigwig/ENCFF480GFU.bigWig	data/bigwig/ENCFF480GFU.bigWig	lambda x: np.arcsinh(np.nan_to_num(x))
data/bigwig/ENCFF815ONV.bigWig	data/bigwig/ENCFF815ONV.bigWig	lambda x: np.arcsinh(np.nan_to_num(x))
```
In addition, create a `.tsv` file listing the genomic positions of interest, with the following columns: `chrom`, `pos`, `strand`.
```tsv
chrom	pos	strand
chr1	11868	1
chr1	12009	1
chr1	29569	-1
chr1	17435	-1
```
After preparing these files, run `preprocess.sh` with the paths to the genome `.fa` file, the profile and position `.tsv` files, and an output folder for writing the generated TFRecord files. For multi-species training, run the preprocessing step separately for each species. Next, run `train.sh`, specifying the TFRecord folder(s) and an output folder for saving the trained model. After training, run `finetune.sh` using the trained model as input. The fine-tuned model will be saved in a new folder with `_finetune` appended to the original model folder name.

## Contact

- Kishore Jaganathan: [kjaganathan@illumina.com](mailto:kjaganathan@illumina.com)  
- Gherman Novakovsky: [gnovakovsky@illumina.com](mailto:gnovakovsky@illumina.com)  
- Kyle Farh: [kfarh@illumina.com](mailto:kfarh@illumina.com)
