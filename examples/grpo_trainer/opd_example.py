#!/usr/bin/env python3
"""
Example: Online Policy Distillation (OPD) with verl

This script demonstrates how to set up OPD training programmatically.
OPD distills a teacher model into a student model using dense per-token feedback.
"""

from omegaconf import DictConfig, OmegaConf


def create_opd_config(
    student_model_path: str = "Qwen/Qwen3-8B",
    teacher_model_path: str = None,  # None = use student as teacher
    train_files: str = "$HOME/data/gsm8k/train.parquet",
    val_files: str = "$HOME/data/gsm8k/test.parquet",
    learning_rate: float = 1e-6,
    batch_size: int = 512,
    n_gpus: int = 8,
) -> DictConfig:
    """
    Create configuration for OPD training.
    
    Args:
        student_model_path: Path to student model (the model being trained)
        teacher_model_path: Path to teacher model (None = use student model as reference)
        train_files: Path to training data
        val_files: Path to validation data
        learning_rate: Learning rate for student model
        batch_size: Global batch size
        n_gpus: Number of GPUs per node
    
    Returns:
        OmegaConf configuration for OPD training
    """
    
    config = {
        # Algorithm: Use OPD advantage estimator
        "algorithm": {
            "adv_estimator": "opd",
            "use_kl_in_reward": False,
            "norm_adv_by_std_in_grpo": False,
            "gamma": 1.0,
            "lam": 1.0,
        },
        
        # Data configuration
        "data": {
            "train_files": train_files,
            "val_files": val_files,
            "train_batch_size": batch_size,
            "max_prompt_length": 512,
            "max_response_length": 1024,
            "filter_overlong_prompts": True,
            "truncation": "error",
        },
        
        # Actor (student model) configuration
        "actor_rollout_ref": {
            "model": {
                "path": student_model_path,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
            
            # Actor training settings
            "actor": {
                "optim": {"lr": learning_rate},
                "ppo_mini_batch_size": 256,
                "ppo_micro_batch_size_per_gpu": 32,
                "use_kl_loss": False,  # No KL loss (implicit in OPD advantages)
                "entropy_coeff": 0.0,
                "fsdp_config": {
                    "param_offload": False,
                    "optimizer_offload": False,
                },
            },
            
            # Rollout settings (student generates trajectories)
            "rollout": {
                "name": "vllm",
                "n": 1,  # OPD works with n=1 (unlike GRPO which needs n>1)
                "tensor_model_parallel_size": 2,
                "gpu_memory_utilization": 0.5,
                "log_prob_micro_batch_size_per_gpu": 32,
            },
            
            # Reference/Teacher policy settings
            # Note: Teacher model is automatically enabled when algorithm.adv_estimator=opd
            "ref": {
                "model": {
                    "path": teacher_model_path or student_model_path,
                },
                "log_prob_micro_batch_size_per_gpu": 32,
                "fsdp_config": {
                    "param_offload": True,  # Offload teacher to save memory
                },
            },
        },
        
        # Trainer settings
        "trainer": {
            "critic_warmup": 0,  # No critic in OPD
            "logger": ["console", "wandb"],
            "project_name": "verl_opd_example",
            "experiment_name": "opd_distillation",
            "n_gpus_per_node": n_gpus,
            "nnodes": 1,
            "save_freq": 20,
            "test_freq": 5,
            "total_epochs": 15,
        },
    }
    
    return OmegaConf.create(config)


def main():
    """Example usage of OPD configuration."""
    
    # Example 1: Self-distillation (student = teacher)
    print("Example 1: Self-distillation (Qwen3-8B with itself as teacher)")
    config_self = create_opd_config(
        student_model_path="Qwen/Qwen3-8B",
        teacher_model_path=None,  # Use same model as reference
    )
    print(OmegaConf.to_yaml(config_self))
    print("\n" + "="*80 + "\n")
    
    # Example 2: Teacher-student distillation (32B -> 8B)
    print("Example 2: Teacher-student distillation (Qwen3-32B -> Qwen3-8B)")
    config_distill = create_opd_config(
        student_model_path="Qwen/Qwen3-8B",
        teacher_model_path="Qwen/Qwen3-32B",
        batch_size=256,  # Smaller batch for larger teacher
    )
    print(OmegaConf.to_yaml(config_distill))
    print("\n" + "="*80 + "\n")
    
    # To use these configs with verl:
    print("To run OPD training:")
    print("1. Save config: OmegaConf.save(config, 'opd_config.yaml')")
    print("2. Run trainer: python -m verl.trainer.main_ppo --config-path . --config-name opd_config")
    print("\nOr use command-line overrides:")
    print("python -m verl.trainer.main_ppo algorithm.adv_estimator=opd ...")


if __name__ == "__main__":
    main()

