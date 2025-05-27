#$ -q gpu-h100
#$ -l gpus=1
#$ -l h_vmem=64G
#$ -cwd
#$ -N score
#$ -e sge/
#$ -o sge/
#$ -j y

model_folder='models/promoterAI_hg38_mm10_finetune'
var_files[0]='data/benchmark/GTEx_outlier.tsv'
var_files[1]='data/benchmark/CAGI5_saturation.tsv'
var_files[2]='data/benchmark/MPRA_saturation.tsv'
var_files[3]='data/benchmark/GTEx_eQTL.tsv'
var_files[4]='data/benchmark/MPRA_eQTL.tsv'
var_files[5]='data/benchmark/UKBB_proteome.tsv'
var_files[6]='data/benchmark/GEL_RNA.tsv'
fasta_file='data/fasta/hg38.fa'

conda activate promoterai
for var_file in ${var_files[*]}
do
    python score.py \
        --model_folder ${model_folder} \
        --var_file ${var_file} \
        --fasta_file ${fasta_file} \
        --input_length 20480
done
conda deactivate
