#$ -l h_vmem=192G
#$ -cwd
#$ -N preprocess
#$ -e sge/
#$ -o sge/
#$ -j y
#$ -t 1-22

tfr_folder='data/tfr_hg38'
tss_file='data/annotation/tss_hg38.tsv'
fasta_file='data/fasta/hg38.fa'
bigwig_files='data/bigwig/hg38.tsv'

conda activate promoterai
python preprocess.py \
    --tfr_folder ${tfr_folder} \
    --tss_file ${tss_file} \
    --fasta_file ${fasta_file} \
    --bigwig_files ${bigwig_files} \
    --chrom 'chr'${SGE_TASK_ID} \
    --input_length 81920 \
    --output_length 16384 \
    --chunk_size 256
conda deactivate
