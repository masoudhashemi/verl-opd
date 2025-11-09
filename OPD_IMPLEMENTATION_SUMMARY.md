# Online Policy Distillation (OPD) Implementation Summary

This document summarizes the implementation of Online Policy Distillation (OPD) in the verl framework, building upon GRPO with minimal changes.

## Overview

**Online Policy Distillation (OPD)** is a training method that distills a teacher model into a student model using dense per-token feedback while maintaining on-policy sampling. It achieves 10-30× compute reduction compared to standard RL while maintaining similar performance.

**Reference**: [Thinking Machines Lab - On-Policy Distillation](https://thinkingmachineslab.com/blog/on-policy-distillation)

## Key Implementation Changes

### 1. Core Algorithm (`verl/trainer/ppo/core_algos.py`)

#### Added OPD Advantage Estimator Enum
```python
class AdvantageEstimator(str, Enum):
    # ... existing estimators ...
    OPD = "opd"  # Online Policy Distillation
```

#### Added `compute_opd_advantage()` Function
- **Location**: Lines 421-513
- **Purpose**: Computes per-token advantages from reverse KL divergence
- **Key Logic**:
  ```python
  reverse_kl = student_log_probs - teacher_log_probs
  advantages = -reverse_kl  # Minimizing this pushes student toward teacher
  ```
- **Inputs**:
  - `teacher_log_probs`: Teacher model's log probabilities for student tokens
  - `student_log_probs`: Student model's log probabilities (from rollout)
  - `response_mask`: Valid token mask
- **Output**: Per-token advantages (dense feedback)

### 2. Trainer Integration (`verl/trainer/ppo/ray_trainer.py`)

#### Added `compute_teacher_log_probs()` Helper Function
- **Location**: Lines 181-215
- **Purpose**: Computes teacher logprobs for student trajectories
- **Reuses**: Reference policy infrastructure (teacher = ref policy)

#### Updated `compute_advantage()` Function
- **Location**: Lines 255-260
- **Change**: Added OPD-specific logprob passing
- **Logic**:
  ```python
  if adv_estimator == core_algos.AdvantageEstimator.OPD:
      if "teacher_log_probs" in data.batch:
          adv_kwargs["teacher_log_probs"] = data.batch["teacher_log_probs"]
      if "rollout_log_probs" in data.batch:
          adv_kwargs["student_log_probs"] = data.batch["rollout_log_probs"]
  ```

#### Added Teacher Logprob Computation in `fit()` Loop
- **Location**: Lines 1190-1204
- **Purpose**: Compute teacher logprobs before computing advantages
- **Logic**:
  ```python
  if self.config.algorithm.adv_estimator == "opd":
      teacher_log_probs = compute_teacher_log_probs(batch, teacher_wg=...)
      batch = batch.union(teacher_log_probs)
  ```

### 3. Example Training Scripts

#### Created Three New Files:

1. **`examples/grpo_trainer/run_qwen3-8b_opd.sh`**
   - Basic OPD training with same model as teacher
   - Key configs: `algorithm.adv_estimator=opd`, `ref.log_prob_estimator_enable=True`

2. **`examples/grpo_trainer/run_qwen3-32b_to_8b_opd.sh`**
   - Teacher-student distillation (32B → 8B)
   - Shows how to use different teacher model

3. **`examples/grpo_trainer/opd_example.py`**
   - Programmatic configuration example
   - Python API usage demonstration

### 4. Documentation

#### Created `examples/grpo_trainer/OPD_README.md`
- Comprehensive documentation (400+ lines)
- Covers:
  - Overview and algorithm
  - Key differences from GRPO
  - Configuration guide
  - Example scripts
  - Hyperparameter recommendations
  - Troubleshooting
  - Mathematical background
  - References

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    OPD Training Loop                        │
└─────────────────────────────────────────────────────────────┘

1. ROLLOUT (Student generates trajectories)
   ├─ Student model (π_θ) generates tokens
   └─ Compute student log_probs (from rollout or recompute)

2. TEACHER EVALUATION (New in OPD)
   ├─ Pass student trajectories to teacher model (π_teacher)
   └─ Compute teacher log_probs for each student token
   
3. ADVANTAGE COMPUTATION (OPD-specific)
   ├─ reverse_KL = log π_θ - log π_teacher  (per token)
   └─ advantages = -reverse_KL  (dense feedback)

4. POLICY UPDATE (Reuses GRPO/PPO infrastructure)
   ├─ Standard policy gradient with computed advantages
   └─ No KL loss needed (implicit in advantages)
```

## Key Differences from GRPO

| Aspect | GRPO | OPD |
|--------|------|-----|
| **Advantage Source** | Group-based outcome rewards | Per-token reverse KL |
| **Teacher Model** | Not used | Required (via ref policy) |
| **Sampling** | Needs n > 1 | Works with n ≥ 1 |
| **Feedback** | Sparse (1 value/sequence) | Dense (per token) |
| **KL Regularization** | `use_kl_loss=True` | `use_kl_loss=False` |
| **Configuration** | `adv_estimator=grpo` | `adv_estimator=opd` |

## Configuration Quick Reference

### Required Changes from GRPO

```yaml
# Set OPD advantage estimator
algorithm:
  adv_estimator: opd
  norm_adv_by_std_in_grpo: false  # Optional, typically false
  use_kl_in_reward: false

# Enable teacher (via reference policy)
actor_rollout_ref:
  ref:
    log_prob_estimator_enable: true  # REQUIRED
    model:
      path: "Qwen/Qwen3-32B"  # Optional: different teacher
  
  # Student settings
  actor:
    use_kl_loss: false  # No KL loss (implicit in advantages)
  
  rollout:
    n: 1  # Can use n=1 (unlike GRPO)
```

## Files Modified/Created

### Modified Files (2)
1. `verl/trainer/ppo/core_algos.py`
   - Added `OPD` to `AdvantageEstimator` enum
   - Added `compute_opd_advantage()` function (~95 lines)

2. `verl/trainer/ppo/ray_trainer.py`
   - Added `compute_teacher_log_probs()` helper (~35 lines)
   - Updated `compute_advantage()` to pass OPD logprobs (~7 lines)
   - Added teacher logprob computation in `fit()` loop (~15 lines)

### Created Files (4)
1. `examples/grpo_trainer/run_qwen3-8b_opd.sh` (basic example)
2. `examples/grpo_trainer/run_qwen3-32b_to_8b_opd.sh` (distillation example)
3. `examples/grpo_trainer/opd_example.py` (programmatic example)
4. `examples/grpo_trainer/OPD_README.md` (comprehensive docs)

**Total**: 2 modified files, 4 new files, ~150 lines of core code, ~500 lines of docs/examples

## Testing Recommendations

### 1. Syntax Check
```bash
python -m py_compile verl/trainer/ppo/core_algos.py
python -m py_compile verl/trainer/ppo/ray_trainer.py
```

### 2. Import Check
```python
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_opd_advantage
from verl.trainer.ppo.ray_trainer import compute_teacher_log_probs
assert AdvantageEstimator.OPD == "opd"
print("✓ OPD imports successful")
```

### 3. Configuration Check
```bash
# Dry-run to check config parsing
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opd \
    actor_rollout_ref.ref.log_prob_estimator_enable=true \
    --cfg job \
    --help
```

### 4. End-to-End Test
```bash
# Run on small data for 1 epoch
bash examples/grpo_trainer/run_qwen3-8b_opd.sh \
    data.train_batch_size=32 \
    trainer.total_epochs=1 \
    trainer.save_freq=999
```

## Usage Examples

### Command Line
```bash
# Basic OPD training
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opd \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.ref.log_prob_estimator_enable=true \
    actor_rollout_ref.actor.use_kl_loss=false

# Teacher-student distillation
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opd \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.ref.model.path=Qwen/Qwen3-32B \
    actor_rollout_ref.ref.log_prob_estimator_enable=true
```

### Python API
```python
from examples.grpo_trainer.opd_example import create_opd_config
config = create_opd_config(
    student_model_path="Qwen/Qwen3-8B",
    teacher_model_path="Qwen/Qwen3-32B"
)
# Use with verl trainer...
```

## Performance Expectations

Based on the original OPD paper:

- **Compute**: 10-30× faster than standard RL
- **Data efficiency**: Works with fewer prompts (dense feedback)
- **Quality**: Comparable to RL, matches teacher behavior
- **Memory**: Similar to GRPO (teacher can be offloaded)

## Potential Extensions

### Future Improvements (Not Implemented)
1. **Adaptive teacher weighting**: Blend multiple teachers
2. **Curriculum learning**: Gradually increase teacher difficulty
3. **Token-level filtering**: Skip tokens where student ≈ teacher
4. **Forward KL option**: Add `kl_type="forward"|"reverse"` config
5. **Multi-teacher distillation**: Average logprobs from multiple teachers

## Troubleshooting

### Common Issues

1. **"OPD requires 'teacher_log_probs'"**
   - **Fix**: Set `actor_rollout_ref.ref.log_prob_estimator_enable=true`

2. **"OPD requires a reference policy"**
   - **Fix**: Same as above, enable reference policy

3. **High GPU memory usage**
   - **Fix**: Offload teacher with `ref.fsdp_config.param_offload=true`

4. **Slow training**
   - **Check**: Is teacher model too large?
   - **Fix**: Use smaller teacher or reduce batch size

## References

1. **Thinking Machines Lab**: [On-Policy Distillation](https://thinkingmachineslab.com/blog/on-policy-distillation)
2. **DeepSeek**: [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300)
3. **DAGGER**: [Dataset Aggregation for Imitation Learning](https://arxiv.org/abs/1011.0686)

## Acknowledgments

This implementation was designed to be minimally invasive, reusing existing GRPO/PPO infrastructure. The key insight is that OPD can be viewed as a special case of policy gradient with per-token dense advantages.

---

**Implementation Date**: 2025-01-09
**Version**: v0.1.0
**Status**: ✅ Ready for testing

