export CUDA_VISIBLE_DEVICES="5"


python  inference.py \
--lora_path lora_model/checkpoint-4250 \
--model_path model/ZhipuAI/chatglm3-6b-32k \
--pub_path  data/IND-WhoIsWho/pid_to_info_all.json \
--eval_path data/IND-WhoIsWho/ind_test_author_filter_public.json \
--saved_dir data/IND-WhoIsWho/sub
