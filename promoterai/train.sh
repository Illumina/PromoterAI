#$ -q gpu-h100
#$ -l gpus=8
#$ -l h_vmem=448G
#$ -cwd
#$ -N train
#$ -e sge/
#$ -o sge/
#$ -j y

model_folder='models/promoterAI_hg38_mm10'
tfr_human_folder='data/tfr_hg38'
tfr_nonhuman_folders[0]='data/tfr_mm10'

conda activate promoterai
python train.py \
    --model_folder ${model_folder} \
    --tfr_human_folder ${tfr_human_folder} \
    --tfr_nonhuman_folders ${tfr_nonhuman_folders[@]} \
    --input_length 20480 \
    --output_length 4096 \
    --num_blocks 24 \
    --model_dim 1024 \
    --batch_size 32
conda deactivate
