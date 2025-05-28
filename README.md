# PromoterAI

This repository contains the source code for PromoterAI, a deep learning model for predicting the impact of promoter variants on gene expression. Scores range from –1 to 1, with negative values indicating under-expression and positive values indicating over-expression.

Precomputed PromoterAI scores for all human promoter single nucleotide variants are freely available for academic and non-commercial research use upon acceptance of the [academic license agreement](https://illumina2.na1.adobesign.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhAuRnD5FtTNwyNo-5X6njTJqQOOMu3V_0nU0MjxSi_9PLCrquWaKSRrT3e1RhHkr7w*).

## Overview

The scripts included in this repository serve as a reference for the PromoterAI training and inference pipeline. Full execution requires additional resources and model weights that are not part of this repository.

To begin, download the appropriate reference genome (`.fa`) and regulatory profile (`.bigWig`) tracks. Organize the `.bigWig` file paths and their corresponding transformations into a `.tsv` file, where each row represents a prediction target, with the following columns:  
- `fwd`: path to the forward-strand `.bigWig` file,  
- `rev`: path to the reverse-strand `.bigWig` file,  
- `xform`: transformation applied to the prediction target.  
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

Once these files are prepared, set up the `promoterai` conda environment (`environment.yml`) and run `preprocess.sh` with the paths to the genome `.fa` file, the profile and position `.tsv` files, and an output folder for writing the generated TFRecord files. Next, run `train.sh`, specifying the TFRecord folder(s) and an output folder for saving the trained model. After training, run `finetune.sh` using the trained model as input. The fine-tuned model will be saved in a new folder with `_finetune` appended to the original model folder name.

To score variants, organize them in a `.tsv` file with the following columns: `chrom`, `pos`, `ref`, `alt`, `strand`.
```tsv
chrom	pos	ref	alt	strand
chr16	84145214	G	T	1
chr16	84145333	G	C	1
chr2	55232249	T	G	-1
chr2	55232374	C	T	-1
```
Run `score.sh` with the path to the genome `.fa` file, the model folder, and the variant `.tsv` file. Scores will be added as a new column labeled `score`, with the output file name constructed by appending the model folder’s basename to the variant file name.

## Contact

Kishore Jaganathan: [kjaganathan@illumina.com](mailto:kjaganathan@illumina.com)  
Gherman Novakovsky: [gnovakovsky@illumina.com](mailto:gnovakovsky@illumina.com)  
Kyle Farh: [kfarh@illumina.com](mailto:kfarh@illumina.com)
