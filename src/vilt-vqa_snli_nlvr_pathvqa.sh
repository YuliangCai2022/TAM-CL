export TOKENIZERS_PARALLELISM=false

python -m run --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks vqa,snli-ve,nlvr2,pathvqa\
                        --cl_algorithm sequential_ft \
                        --climb_data_dir /project/rostamim_919/caiyulia/data/ \
            		    --do_train \
                        --do_eval \
                        --output_dir /project/rostamim_919/caiyulia/Multi-Dytox/output/ \
                        --batch_size 16 \
                        --task_attention 1 \
                        --dytox 1 \
                        --ewc 0 \
                        --parallel 0 \
                        --replay 0
                        #--do_wandb_logging 
'''

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run.py --encoder_name vilt \
                        --pretrained_model_name dandelin/vilt-b32-mlm \
                        --ordered_cl_tasks vqa,snli-ve,nlvr2\
                        --cl_algorithm sequential_ft \
                        --climb_data_dir /project/rostamim_919/caiyulia/data/ \
            		    --do_train \
                        --do_eval \
                        --output_dir /project/rostamim_919/caiyulia/Multi-Dytox/output/ \
                        --batch_size 8 \
                        --task_attention 1 \
                        --dytox 1 \
                        --ewc 0 \
                        --parallel 1
                        #--do_wandb_logging 
                        
'''