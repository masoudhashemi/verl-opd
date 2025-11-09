# Where the Teacher Model is Loaded in OPD

## Quick Answer

The **teacher model is loaded in `verl/workers/fsdp_workers.py`, lines 814-838**, inside the `ActorRolloutRefWorker.init_model()` method.

---

## Complete Loading Flow

### 1. Configuration (User Level)

The teacher model is specified in your configuration:

```bash
actor_rollout_ref.ref.log_prob_estimator_enable=True  # Enable reference policy
actor_rollout_ref.ref.model.path=Qwen/Qwen3-32B        # Teacher model path
```

**Key config paths**:
- `actor_rollout_ref.ref.log_prob_estimator_enable=True` → Enables reference policy as teacher
- `actor_rollout_ref.ref.model.path=<HuggingFace-Path>` → Specifies teacher model

### 2. Trainer Initialization (`verl/trainer/ppo/ray_trainer.py`)

**Lines 756-764**: Reference policy worker is created

```python
# create reference policy if needed
if self.use_reference_policy:
    resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
    ref_policy_cls = RayClassWithInitArgs(
        self.role_worker_mapping[Role.RefPolicy],
        config=self.config.actor_rollout_ref,  # Pass actor_rollout_ref config
        role=str(Role.RefPolicy),
    )
    self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls
```

**Lines 809-811**: Reference policy worker is initialized

```python
if self.use_reference_policy and not self.ref_in_actor:
    self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
    self.ref_policy_wg.init_model()  # ← This triggers model loading
```

### 3. Model Loading (`verl/workers/fsdp_workers.py`)

**Lines 814-838**: The actual teacher model loading happens here

```python
if self._is_ref:
    # Get teacher model path
    ref_model_path = self.config.model.path  # Default: use student path
    ref_model = self.config.ref.get("model", None)
    if ref_model is not None:
        # Override with teacher-specific path if provided
        ref_model_path = ref_model.get("path", self.config.model.path)
    
    if self.rank == 0:
        print("reference model:", ref_model_path)  # ← You'll see this in logs!
    
    # Download/copy model to local path
    local_path = copy_to_local(ref_model_path, use_shm=use_shm)
    
    # Build reference model with FSDP
    self.ref_module_fsdp = self._build_model_optimizer(
        model_path=local_path,
        fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
        optim_config=None,  # No optimizer for teacher (inference only)
        override_model_config=override_model_config,
        use_remove_padding=use_remove_padding,
        use_fused_kernels=use_fused_kernels,
        trust_remote_code=self.config.model.get("trust_remote_code", False),
        use_liger=self.config.model.get("use_liger", False),
        role="ref",  # ← Identifies this as reference/teacher model
    )[0]
    
    # Wrap in DataParallelPPOActor for inference
    self.ref_policy = DataParallelPPOActor(
        config=self.config.ref, 
        actor_module=self.ref_module_fsdp
    )
```

### 4. Model Building (`verl/workers/fsdp_workers.py`)

**Lines 269-550**: `_build_model_optimizer()` is called

```python
def _build_model_optimizer(
    self,
    model_path,  # ← local_path from above (e.g., /tmp/Qwen3-32B)
    fsdp_config: FSDPEngineConfig,
    optim_config,
    override_model_config,
    use_remove_padding=False,
    use_fused_kernels=False,
    enable_gradient_checkpointing=False,
    trust_remote_code=False,
    use_liger=False,
    role="actor",  # ← "ref" for teacher model
    enable_activation_offload=False,
):
    # ...
    
    # Load tokenizer
    self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
    
    # ...
    
    # Load model from HuggingFace
    if self.config.model.get("multimodal", False):
        actor_module = AutoModelForVision2Seq.from_pretrained(
            local_path,
            config=hf_config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32,  # Will be converted to bf16 later
        )
    else:
        actor_module = AutoModelForCausalLM.from_pretrained(
            local_path,  # ← HuggingFace downloads/loads model here!
            config=hf_config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32,
        )
    
    # Wrap with FSDP for distributed inference
    actor_fsdp = FSDP(
        actor_module,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        # ...
    )
    
    return actor_fsdp, None, None, hf_config  # optimizer=None for ref
```

### 5. Usage During Training (`verl/trainer/ppo/ray_trainer.py`)

**Lines 1211-1217**: Teacher model is used for OPD

```python
# For OPD, compute teacher log probs (reuse reference policy as teacher)
if self.config.algorithm.adv_estimator == "opd":
    with marked_timer("teacher_log_prob", timing_raw, color="purple"):
        if not self.use_reference_policy:
            raise ValueError(
                "OPD requires a reference policy to act as teacher. "
                "Please set actor_rollout_ref.ref.log_prob_estimator_enable=true"
            )
        # Compute teacher logprobs (reuse ref policy infrastructure)
        teacher_log_probs = compute_teacher_log_probs(
            batch, 
            teacher_wg=self.ref_policy_wg,  # ← Uses the loaded teacher model
            ref_in_actor=self.ref_in_actor
        )
        batch = batch.union(teacher_log_probs)
```

---

## Key Files & Line Numbers

| File | Lines | Purpose |
|------|-------|---------|
| `verl/trainer/ppo/ray_trainer.py` | 756-764 | Create reference policy worker class |
| `verl/trainer/ppo/ray_trainer.py` | 809-811 | Initialize reference policy worker |
| `verl/trainer/ppo/ray_trainer.py` | 1211-1217 | Use teacher for OPD log probs |
| `verl/workers/fsdp_workers.py` | 814-838 | **Main teacher loading logic** |
| `verl/workers/fsdp_workers.py` | 269-550 | Build model with HuggingFace + FSDP |

---

## Configuration Hierarchy

```
actor_rollout_ref:              # Shared config for actor/rollout/ref
  model:
    path: "Qwen/Qwen3-8B"       # Student model path
  
  ref:                          # Teacher-specific config
    log_prob_estimator_enable: true   # ← REQUIRED for OPD
    model:
      path: "Qwen/Qwen3-32B"    # ← Teacher model path (overrides default)
    fsdp_config:
      param_offload: true       # Offload teacher to save memory
      optimizer_offload: true
    log_prob_micro_batch_size_per_gpu: 8
```

**Path resolution logic**:
1. If `actor_rollout_ref.ref.model.path` is specified → use that (different teacher)
2. Otherwise → use `actor_rollout_ref.model.path` (same model as student)

---

## How to Verify Teacher Model is Loaded

### 1. Check Training Logs

When training starts, you'll see:

```bash
reference model: Qwen/Qwen3-32B
```

This is printed at line 821 in `fsdp_workers.py`.

### 2. Check Memory Usage

Teacher model loading increases GPU memory:

```bash
# Before ref model load
[Before init ref from HF AutoModel] GPU memory: 12.3 GB

# After ref model load  
[After init ref from HF AutoModel] GPU memory: 45.8 GB
```

### 3. Check Worker Initialization

In Ray dashboard or logs, you'll see:

```bash
Creating worker group for role: ref
Initializing RefPolicy worker with 8 GPUs
Loading model from: /tmp/ray/session_*/Qwen3-32B/...
```

---

## Common Configurations

### Same Model as Teacher (Self-Distillation)

```bash
actor_rollout_ref.model.path=Qwen/Qwen3-8B
actor_rollout_ref.ref.log_prob_estimator_enable=True
# No ref.model.path → uses same model as student
```

### Different Model as Teacher (Knowledge Distillation)

```bash
actor_rollout_ref.model.path=Qwen/Qwen3-8B          # Student (small)
actor_rollout_ref.ref.model.path=Qwen/Qwen3-32B     # Teacher (large)
actor_rollout_ref.ref.log_prob_estimator_enable=True
```

### Memory-Efficient Teacher

```bash
actor_rollout_ref.ref.model.path=Qwen/Qwen3-32B
actor_rollout_ref.ref.log_prob_estimator_enable=True
actor_rollout_ref.ref.fsdp_config.param_offload=True    # Offload to CPU
actor_rollout_ref.ref.fsdp_config.optimizer_offload=True
```

---

## Summary

**Teacher model is loaded at**:
- **File**: `verl/workers/fsdp_workers.py`
- **Function**: `ActorRolloutRefWorker.init_model()`
- **Lines**: 814-838
- **Trigger**: `self.ref_policy_wg.init_model()` in trainer (line 811 of `ray_trainer.py`)
- **Config**: `actor_rollout_ref.ref.model.path`

The teacher model is:
1. ✅ Loaded from HuggingFace (or local path)
2. ✅ Wrapped with FSDP for distributed inference
3. ✅ Optionally offloaded to CPU to save memory
4. ✅ Used only for inference (no optimizer, no gradients)
5. ✅ Accessed via `self.ref_policy_wg` in the trainer

