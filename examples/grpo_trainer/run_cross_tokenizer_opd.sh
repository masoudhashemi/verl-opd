#!/bin/bash
# Cross-Tokenizer Online Policy Distillation (OPD)
# 
# This example demonstrates distilling across different model architectures
# with different tokenizers (e.g., GPT-style → LLaMA-style).
#
# Text splitting enables cross-tokenizer distillation by:
# 1. Decoding student tokens to text
# 2. Splitting text into coarser units (words, sentences, etc.)
# 3. Aggregating logprobs within each split
# 4. Computing KL at the split level
# 5. Broadcasting split-level advantages back to tokens
#
# This works even when teacher and student have completely different tokenizers!

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opd \
    algorithm.opd_text_splitter.name=word \
    algorithm.opd_text_splitter.aggregation=mean \
    algorithm.opd_text_splitter.params.keep_whitespace=true \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=meta-llama/Llama-3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_estimator_enable=True \
    actor_rollout_ref.ref.model.path=Qwen/Qwen3-32B \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_cross_tokenizer_opd' \
    trainer.experiment_name='llama3_8b_from_qwen3_32b' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 $@


