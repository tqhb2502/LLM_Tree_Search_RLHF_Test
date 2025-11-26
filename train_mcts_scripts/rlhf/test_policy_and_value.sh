set -e
# export TEST_NO_TERMINAL=1
export TEST_WITH_TERMINAL=1
# export TEST_COT_GREEDY=1
# export TEST_COT_SC=1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CT2_DIR=/content/LLM_Tree_Search_RLHF_Test/ct2_cache/gpt2
CRITIC_PATH=/content/LLM_Tree_Search_RLHF_Test/value_models/gpt2-open-instruct-v1/last_model_hf
torchrun --nproc_per_node=1 --master-port 29503 /content/LLM_Tree_Search_RLHF_Test/tsllm/offline_rl/test_sft_and_v_rlhf.py \
    --critic_model_path $CRITIC_PATH \
    --tokenizer_path $CRITIC_PATH \
    --ct2_dir $CT2_DIR \
    --save_dir $1/policy_ep1 \
    --env_name rlhf
