export TOKENIZERS_PARALLELISM=false

python -m run --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks gqa\
                        --cl_algorithm sequential_ft \
                        --climb_data_dir /project/rostamim_919/caiyulia/data/ \
            		    --do_train \
                        --do_eval \
                        --output_dir /project/rostamim_919/caiyulia/Multi-Dytox/output/ \
                        --batch_size 8 \
                        --task_attention 1 \
                        --dytox 1\
                        --ewc 0 \
                        --parallel 0 \
                        --replay 1