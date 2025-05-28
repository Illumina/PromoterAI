#$ -q gpu-h100
#$ -l gpus=2
#$ -l h_vmem=128G
#$ -cwd
#$ -N finetune
#$ -e sge/
#$ -o sge/
#$ -j y

model_folder='models/promoterAI_hg38_mm10'
var_file='data/annotation/finetune_gtex.tsv'
fasta_file='data/fasta/hg38.fa'

conda activate promoterai
python finetune.py \
    --model_folder ${model_folder} \
    --var_file ${var_file} \
    --fasta_file ${fasta_file} \
    --input_length 20480 \
    --batch_size 8
conda deactivate
