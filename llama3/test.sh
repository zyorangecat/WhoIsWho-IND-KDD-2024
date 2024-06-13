CUDA_VISIBLE_DEVICES=5 python  inference.py \
--lora_path llama3_lora_model \
--model_path  model/LLM-Research/Meta_Llama_3_8B_Instruct \
--pub_path  data/IND-WhoIsWho/pid_to_info_all.json \
--eval_path /data/IND-WhoIsWho/ind_test_author_filter_public.json \
--saved_dir data/IND-WhoIsWho/sub

