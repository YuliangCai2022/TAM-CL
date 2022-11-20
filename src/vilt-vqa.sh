export TOKENIZERS_PARALLELISM=false

python -m run --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks vqa \
                        --cl_algorithm singletask_ft \
                        --climb_data_dir /project/rostamim_919/caiyulia/data/ \
            		--do_train \
                        --do_eval \
                        --output_dir /project/rostamim_919/caiyulia/Multi-Dytox/output/ \
                        --batch_size 32 \
                        --task_attention 0 \
                        --do_wandb_logging \
                        --dytox 0
