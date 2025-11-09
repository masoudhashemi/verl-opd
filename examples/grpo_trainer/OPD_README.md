# Online Policy Distillation (OPD)

Online Policy Distillation is a training paradigm that combines the advantages of on-policy reinforcement learning with supervised distillation to efficiently train smaller or specialized models from larger teacher models.

## Overview

OPD achieves **10-30× compute reduction** compared to standard RL while maintaining similar performance by:

1. **On-policy sampling**: Generate trajectories from the student model (not the teacher)
2. **Dense supervision**: Compute per-token feedback using reverse KL divergence
3. **Teacher guidance**: Use a teacher model to grade each token in the student's trajectories

Unlike traditional methods:
- **vs SFT/Distillation**: OPD is on-policy (student samples its own data), avoiding distribution mismatch
- **vs RL (GRPO/PPO)**: OPD uses dense per-token rewards instead of sparse scalar rewards
- **vs GRPO**: OPD uses teacher logprobs for advantages instead of group-based reward normalization

## How It Works

### Algorithm

```
1. Sample trajectories from student model π_θ
2. For each token in student trajectory:
   - Compute student log prob: log π_θ(token|context)
   - Compute teacher log prob: log π_teacher(token|context)
   - Compute reverse KL: reverse_KL = log π_θ - log π_teacher
3. Use -reverse_KL as per-token advantages
4. Update student with policy gradient (reusing PPO/GRPO infrastructure)
```

### Key Differences from GRPO

| Aspect | GRPO | OPD |
|--------|------|-----|
| **Advantage Source** | Group-based outcome rewards | Per-token reverse KL |
| **Feedback Density** | Sparse (scalar per sequence) | Dense (per token) |
| **Teacher Model** | Not used | Required |
| **Sampling** | n > 1 (group sampling) | n ≥ 1 (works with n=1) |
| **KL Regularization** | Via kl_loss in actor | Implicit in advantages |

## Configuration

### Required Changes from GRPO

```bash
# Set advantage estimator to OPD
algorithm.adv_estimator=opd

# Teacher model is automatically enabled when using OPD
# (No additional config needed - algorithm.adv_estimator=opd enables it)

# Disable KL loss (advantages already encode teacher preference)
actor_rollout_ref.actor.use_kl_loss=False

# Disable advantage normalization (optional, typically False for OPD)
algorithm.norm_adv_by_std_in_grpo=False

# Can use n=1 (unlike GRPO which needs n>1 for group sampling)
actor_rollout_ref.rollout.n=1
```

### Optional: Using a Different Teacher Model

By default, OPD uses the reference policy as the teacher. To use a larger/different teacher model:

```bash
# Student model
actor_rollout_ref.model.path=Qwen/Qwen3-8B

# Teacher model (via reference policy)
actor_rollout_ref.ref.model.path=Qwen/Qwen3-32B
```

## Example Scripts

### Basic OPD Training

Train Qwen3-8B using itself as the teacher (same as reference policy):

```bash
bash examples/grpo_trainer/run_qwen3-8b_opd.sh
```

### Teacher-Student Distillation

Distill Qwen3-32B (teacher) → Qwen3-8B (student):

```bash
bash examples/grpo_trainer/run_qwen3-32b_to_8b_opd.sh
```

## Implementation Details

### Code Structure

The OPD implementation reuses GRPO/PPO infrastructure with minimal changes:

1. **Advantage Estimator** (`verl/trainer/ppo/core_algos.py`):
   - `compute_opd_advantage()`: Computes per-token advantages from reverse KL

2. **Teacher Logprob Computation** (`verl/trainer/ppo/ray_trainer.py`):
   - `compute_teacher_log_probs()`: Gets teacher logprobs for student trajectories
   - Integrated into training loop before advantage computation

3. **Training Scripts** (`examples/grpo_trainer/`):
   - `run_qwen3-8b_opd.sh`: Basic OPD example
   - `run_qwen3-32b_to_8b_opd.sh`: Teacher-student distillation example

### Reverse KL Divergence

OPD uses **reverse KL** (mode-seeking) instead of forward KL:

```
reverse_KL(π_student || π_teacher) = E[log π_student - log π_teacher]
advantages = -reverse_KL
```

**Why reverse KL?**
- **Mode-seeking**: Focuses on teacher's best behavior, not averaging multiple modes
- **Unhackable**: Low KL always means closer to teacher (unlike scalar rewards)
- **Efficient**: Can be computed per-token without full trajectory

## Performance Characteristics

### Compute Efficiency

- **10-30× faster** than standard RL (PPO/GRPO)
- **Dense feedback**: O(N) bits per sequence vs O(1) for RL
- **Minimal sampling**: Works with n=1 (vs n>>1 for GRPO)

### When to Use OPD

**Good for:**
- Distilling larger teacher models into smaller students
- Recovering lost capabilities after domain fine-tuning
- Data-efficient learning (works with few prompts)
- Fast iteration (lower compute than RL)

**Not ideal for:**
- No teacher model available
- Teacher is not better than student
- Need exploration beyond teacher capabilities

## Hyperparameter Recommendations

Based on the original OPD paper and experiments:

```yaml
# Learning rate: similar to GRPO
actor_rollout_ref.actor.optim.lr: 1e-6

# No KL loss (implicit in advantages)
actor_rollout_ref.actor.use_kl_loss: False

# No advantage normalization (optional)
algorithm.norm_adv_by_std_in_grpo: False

# Can use n=1 for efficiency
actor_rollout_ref.rollout.n: 1

# Loss aggregation: token-mean works well
actor_rollout_ref.actor.loss_agg_mode: "token-mean"
```

## Troubleshooting

### Error: "OPD requires 'teacher_log_probs' in kwargs"

**Cause**: Teacher logprobs not computed before advantage estimation.

**Solution**: This should be automatically enabled when `algorithm.adv_estimator=opd`.
If you see this error, ensure you're using the latest version of the code.

### Error: "OPD requires a reference policy to act as teacher"

**Cause**: Reference policy (teacher) not created. This should not happen with correct OPD configuration.

**Solution**: Enable reference policy as shown above.

### Low performance

**Check:**
1. Teacher model is actually better than student
2. Learning rate not too high/low
3. Sufficient training data
4. Teacher model path is correct

## Mathematical Background

### Objective

OPD minimizes the reverse KL divergence between student and teacher:

```
minimize E_x ~ π_θ [KL(π_θ || π_teacher)]
       = E_x ~ π_θ [log π_θ(x) - log π_teacher(x)]
```

### Gradient

Using the log-derivative trick:

```
∇_θ E[KL] = E[∇_θ log π_θ(x) * (log π_θ(x) - log π_teacher(x))]
```

Setting `advantages = -(log π_θ - log π_teacher)`:

```
∇_θ E[KL] = -E[∇_θ log π_θ(x) * advantages]
```

This is exactly the **policy gradient formula** used in RL!

### Connection to DAGGER

OPD is related to DAGGER (Dataset Aggregation):
- Both use on-policy sampling (student generates data)
- Both use teacher feedback (DAGGER: actions, OPD: logprobs)
- OPD is online (no dataset aggregation needed)

## References

1. **On-Policy Distillation** (Thinking Machines Lab)
   - https://thinkingmachineslab.com/blog/on-policy-distillation

2. **DeepSeekMath** (GRPO paper)
   - https://arxiv.org/abs/2402.03300

3. **DAGGER** (Dataset Aggregation)
   - Ross et al., 2010

4. **Let's Verify Step by Step** (Process Rewards)
   - OpenAI, 2023

## Citation

If you use OPD in your research, please cite:

```bibtex
@article{opd2025,
  title={On-Policy Distillation},
  author={Thinking Machines Lab},
  year={2025},
  url={https://thinkingmachineslab.com/blog/on-policy-distillation}
}
```

