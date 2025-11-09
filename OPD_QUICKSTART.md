# Online Policy Distillation (OPD) - Quick Start Guide

🎉 **OPD has been successfully implemented in verl!**

## What is OPD?

Online Policy Distillation (OPD) is a training method that achieves **10-30× compute reduction** vs standard RL by:
- Sampling from student model (on-policy)
- Using teacher model for dense per-token feedback (reverse KL)
- Training with standard policy gradient infrastructure

**Blog post**: https://thinkingmachineslab.com/blog/on-policy-distillation

## Quick Start

### 1. Basic OPD Training (Self-Distillation)

Train a model using itself as the teacher:

```bash
bash examples/grpo_trainer/run_qwen3-8b_opd.sh
```

### 2. Teacher-Student Distillation

Distill a larger teacher (32B) into a smaller student (8B):

```bash
bash examples/grpo_trainer/run_qwen3-32b_to_8b_opd.sh
```

### 3. Custom Configuration

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opd \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.ref.model.path=Qwen/Qwen3-32B \
    actor_rollout_ref.actor.use_kl_loss=false \
    data.train_files=$HOME/data/gsm8k/train.parquet
```

## Key Configuration Changes from GRPO

| Setting | GRPO | OPD |
|---------|------|-----|
| `algorithm.adv_estimator` | `grpo` | `opd` ⚠️ |
| `actor_rollout_ref.actor.use_kl_loss` | `true` | `false` |
| `actor_rollout_ref.rollout.n` | `>1` (e.g., 5) | `≥1` (can use 1) |
| `algorithm.norm_adv_by_std_in_grpo` | `true` | `false` |
| `algorithm.use_kl_in_reward` | varies | `false` |

⚠️ = **Setting this automatically enables teacher model**

## Documentation

- **Full README**: [`examples/grpo_trainer/OPD_README.md`](examples/grpo_trainer/OPD_README.md)
- **Implementation Details**: [`OPD_IMPLEMENTATION_SUMMARY.md`](OPD_IMPLEMENTATION_SUMMARY.md)
- **Python API Example**: [`examples/grpo_trainer/opd_example.py`](examples/grpo_trainer/opd_example.py)

## Verification

Run the verification script to check your setup:

```bash
python3 verify_opd_implementation.py
```

Expected output:
```
✅ All checks passed! OPD implementation is complete.
```

## Testing

Run unit tests:

```bash
python -m pytest tests/trainer/test_opd_advantage.py -v
```

Or run standalone:

```bash
python3 tests/trainer/test_opd_advantage.py
```

## Files Modified/Created

### Core Implementation (2 modified files)
- ✏️ `verl/trainer/ppo/core_algos.py` - Added OPD advantage estimator
- ✏️ `verl/trainer/ppo/ray_trainer.py` - Added teacher logprob computation

### Examples & Documentation (6 new files)
- 📄 `examples/grpo_trainer/run_qwen3-8b_opd.sh`
- 📄 `examples/grpo_trainer/run_qwen3-32b_to_8b_opd.sh`
- 📄 `examples/grpo_trainer/opd_example.py`
- 📄 `examples/grpo_trainer/OPD_README.md`
- 📄 `OPD_IMPLEMENTATION_SUMMARY.md`
- 📄 `OPD_QUICKSTART.md` (this file)

### Testing & Verification (2 new files)
- 🧪 `tests/trainer/test_opd_advantage.py`
- 🔍 `verify_opd_implementation.py`

## How It Works

```
┌──────────────────────────────────────────────────────┐
│  1. Student generates trajectories (on-policy)       │
│     π_θ(tokens | prompts) → student_log_probs       │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  2. Teacher evaluates student's tokens               │
│     π_teacher(tokens | prompts) → teacher_log_probs │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  3. Compute reverse KL per token                     │
│     reverse_KL = student_log_probs - teacher_log_probs│
│     advantages = -reverse_KL                         │
└──────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────┐
│  4. Update student with policy gradient              │
│     (reuses existing GRPO/PPO infrastructure)        │
└──────────────────────────────────────────────────────┘
```

## Troubleshooting

### Error: "OPD requires a reference policy to act as teacher"

**Solution**: This should be automatically enabled when using `algorithm.adv_estimator=opd`.
If you see this error, ensure you're using the latest version of the code.

### Error: "OPD requires 'teacher_log_probs' in kwargs"

**Solution**: Same as above - reference policy must be enabled.

### High GPU memory usage

**Solution**: Offload teacher model:
```bash
actor_rollout_ref.ref.fsdp_config.param_offload=true
actor_rollout_ref.ref.fsdp_config.optimizer_offload=true
```

## Performance Tips

1. **Use n=1**: Unlike GRPO, OPD works well with single sampling (`rollout.n=1`)
2. **Offload teacher**: Set `ref.fsdp_config.param_offload=true` to save memory
3. **Disable KL loss**: Set `actor.use_kl_loss=false` (advantages already encode teacher preference)
4. **Adjust batch size**: Use smaller batches if teacher model is large

## Next Steps

1. ✅ Verify installation: `python3 verify_opd_implementation.py`
2. 📖 Read full docs: `cat examples/grpo_trainer/OPD_README.md`
3. 🚀 Run example: `bash examples/grpo_trainer/run_qwen3-8b_opd.sh`
4. 🔬 Run tests: `python -m pytest tests/trainer/test_opd_advantage.py`

## Support

- **Full README**: [`examples/grpo_trainer/OPD_README.md`](examples/grpo_trainer/OPD_README.md)
- **Implementation Details**: [`OPD_IMPLEMENTATION_SUMMARY.md`](OPD_IMPLEMENTATION_SUMMARY.md)
- **Original Blog**: https://thinkingmachineslab.com/blog/on-policy-distillation

---

**Status**: ✅ Ready to use  
**Version**: v0.1.0  
**Date**: 2025-01-09

