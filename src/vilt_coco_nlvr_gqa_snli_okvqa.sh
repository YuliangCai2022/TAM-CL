#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=16GB

mamba init bash
source ~/.bashrc

conda activate climb

export TOKENIZERS_PARALLELISM=false

python -m run --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks cocoqa,nlvr2,gqa,snli-ve,okvqa\
                        --cl_algorithm sequential_ft \
                        --climb_data_dir /project/rostamim_919/caiyulia/data/ \
            		    --do_train \
                        --do_eval \
                        --output_dir /project/rostamim_919/caiyulia/Multi-Dytox/output/ \
                        --batch_size 8 \
                        --task_attention 1 \
                        --dytox 1 \
                        --ewc 0 \
                        --parallel 0 \
                        --replay 1