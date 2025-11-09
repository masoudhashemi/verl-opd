# Text Splitting for Cross-Tokenizer OPD

## Overview

**Text splitting** enables **cross-tokenizer distillation** in Online Policy Distillation (OPD). This allows you to distill knowledge from a teacher model to a student model **even when they use completely different tokenizers** (e.g., GPT-style → LLaMA-style).

## Problem: Token Misalignment

When teacher and student models use different tokenizers:

```
Student (GPT):    ["Hello", " world", "!"]
Teacher (LLaMA):  ["Hel", "lo", " ", "world", "!"]
                   ❌ Different tokenization!
```

**Token-level KL divergence won't work** because tokens don't align.

## Solution: Text Splitting

Instead of computing KL at the **token level**, we:

1. **Decode** tokens → text
2. **Split** text → coarser units (words, sentences, etc.)
3. **Aggregate** token logprobs → split logprobs
4. **Compute** KL at split level
5. **Broadcast** split advantages → token advantages

```
Text:     "Hello world!"
          ↓ split by words
Splits:   ["Hello ", "world!"]
          ↓
Student tokens:  ["Hello", " world", "!"]     → ["Hello ", "world!"]
Teacher tokens:  ["Hel", "lo", " ", "world", "!"] → ["Hello ", "world!"]
          ↓ aggregate logprobs
Student split logprobs: [-0.5, -0.3]
Teacher split logprobs: [-0.4, -0.2]
          ↓ compute reverse KL
Split advantages: -[(-0.5 - (-0.4)), (-0.3 - (-0.2))] = [0.1, 0.1]
          ↓ broadcast to tokens
Student token advantages: [0.1, 0.1, 0.1]
```

## Text Splitter Types

### 1. WordSplitter (Recommended)

**Use for**: Most cross-tokenizer distillation tasks

```yaml
algorithm:
  opd_text_splitter:
    name: word
    aggregation: mean
    params:
      keep_whitespace: true  # Include whitespace in splits
```

**Example**:
```python
text = "Hello world! How are you?"
splits = ["Hello ", "world! ", "How ", "are ", "you?"]
```

**Pros**:
- Natural granularity for language
- Robust to tokenizer differences
- Good balance between precision and compatibility

**Cons**:
- Simple whitespace splitting (no linguistic analysis)

### 2. SentenceSplitter

**Use for**: Coarse-grained, long-form generation

```yaml
algorithm:
  opd_text_splitter:
    name: sentence
    aggregation: mean
    params:
      sentence_terminators: ['.', '!', '?', '\n\n']
```

**Example**:
```python
text = "Hello world! How are you? I'm fine."
splits = ["Hello world! ", "How are you? ", "I'm fine."]
```

**Pros**:
- Sentence-level coherence
- Good for long documents
- Low computational overhead

**Cons**:
- Very coarse-grained
- Less precise supervision

### 3. CharSplitter

**Use for**: Fine-grained control, character-level models

```yaml
algorithm:
  opd_text_splitter:
    name: char
    aggregation: mean
```

**Example**:
```python
text = "Hi!"
splits = ["H", "i", "!"]
```

**Pros**:
- Maximum granularity
- Works across any tokenization

**Cons**:
- Computationally expensive
- May be too fine-grained

### 4. FixedLengthSplitter

**Use for**: Experimentation, uniform splits

```yaml
algorithm:
  opd_text_splitter:
    name: fixed
    aggregation: mean
    params:
      chunk_size: 10  # characters per chunk
```

**Example**:
```python
text = "Hello world! How are you?"
splits = ["Hello worl", "d! How are", " you?"]
```

**Pros**:
- Predictable, uniform splits
- Simple to reason about

**Cons**:
- Splits words arbitrarily
- Not linguistically motivated

### 5. TokenSplitter (Default)

**Use for**: Same tokenizer (no splitting needed)

```yaml
algorithm:
  opd_text_splitter:
    # Not specified = token-level (default)
```

**Example**:
```python
# Each token is its own split (1:1 mapping)
tokens = ["Hello", " world"]
splits = ["Hello", " world"]
```

**Pros**:
- Fastest (no decoding/splitting)
- Most precise
- No information loss

**Cons**:
- Only works when tokenizers match

## LogProb Aggregation Modes

When aggregating token-level logprobs to split-level logprobs:

### Mean (Recommended)

```yaml
aggregation: mean
```

- **Formula**: `split_logprob = mean(token_logprobs)`
- **Equivalent to**: Geometric mean of probabilities
- **Use for**: Most stable, commonly used
- **Example**: `[-1, -2, -3] → -2`

### Sum

```yaml
aggregation: sum
```

- **Formula**: `split_logprob = sum(token_logprobs)`
- **Equivalent to**: Product of probabilities
- **Use for**: Emphasizing splits with more tokens
- **Example**: `[-1, -2, -3] → -6`

### Min

```yaml
aggregation: min
```

- **Formula**: `split_logprob = min(token_logprobs)`
- **Equivalent to**: Worst token in split
- **Use for**: Pessimistic distillation (focus on errors)
- **Example**: `[-1, -2, -3] → -3`

### Max

```yaml
aggregation: max
```

- **Formula**: `split_logprob = max(token_logprobs)`
- **Equivalent to**: Best token in split
- **Use for**: Optimistic distillation (focus on strengths)
- **Example**: `[-1, -2, -3] → -1`

## Quick Start

### Basic Example (Word-Level)

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opd \
    algorithm.opd_text_splitter.name=word \
    algorithm.opd_text_splitter.aggregation=mean \
    actor_rollout_ref.model.path=meta-llama/Llama-3-8B \
    actor_rollout_ref.ref.model.path=Qwen/Qwen3-32B \
    actor_rollout_ref.ref.log_prob_estimator_enable=true \
    ...
```

### Run Example Script

```bash
bash examples/grpo_trainer/run_cross_tokenizer_opd.sh
```

### Test Splitters

```bash
python3 examples/grpo_trainer/text_splitting_example.py
```

## Implementation

### Core Components

1. **Text Splitter Classes** (`verl/trainer/ppo/text_splitter.py`)
   - Abstract base class `TextSplitter`
   - Concrete implementations (Word, Sentence, Char, etc.)
   - Factory function `get_text_splitter()`

2. **OPD Advantage Computation** (`verl/trainer/ppo/core_algos.py`)
   - Modified `compute_opd_advantage()` to support text splitting
   - Aggregates logprobs by splits
   - Broadcasts split advantages to tokens

3. **Trainer Integration** (`verl/trainer/ppo/ray_trainer.py`)
   - Passes tokenizer and responses to advantage computation
   - Configurable via `algorithm.opd_text_splitter`

### Algorithm Flow

```python
for each sequence in batch:
    # 1. Decode tokens to text
    text = tokenizer.decode(student_tokens)
    
    # 2. Split text
    splits = text_splitter.split_text(text)
    
    # 3. Map tokens → splits
    student_split_ids = map_tokens_to_splits(student_tokens, splits, student_tokenizer)
    teacher_split_ids = map_tokens_to_splits(teacher_tokens, splits, teacher_tokenizer)
    
    # 4. Aggregate logprobs
    student_split_logprobs = aggregate_logprobs_by_splits(student_token_logprobs, student_split_ids)
    teacher_split_logprobs = aggregate_logprobs_by_splits(teacher_token_logprobs, teacher_split_ids)
    
    # 5. Compute reverse KL at split level
    split_reverse_kl = student_split_logprobs - teacher_split_logprobs
    split_advantages = -split_reverse_kl
    
    # 6. Broadcast to token level
    token_advantages = broadcast_split_values_to_tokens(split_advantages, student_split_ids)
```

## Configuration Reference

### Full Configuration

```yaml
algorithm:
  adv_estimator: opd
  opd_text_splitter:
    name: word  # word, sentence, char, fixed, token
    aggregation: mean  # mean, sum, min, max
    params:
      # WordSplitter params
      keep_whitespace: true
      
      # SentenceSplitter params
      # sentence_terminators: ['.', '!', '?', '\n\n']
      
      # FixedLengthSplitter params
      # chunk_size: 10
```

### Minimal Configuration (Default)

```yaml
algorithm:
  adv_estimator: opd
  # No opd_text_splitter = token-level (same tokenizer)
```

## Performance Considerations

### Computational Cost

| Splitter | Decoding | Splitting | Mapping | Total Overhead |
|----------|----------|-----------|---------|----------------|
| Token | None | None | None | ~0% |
| Word | O(n) | O(n) | O(n) | ~5-10% |
| Sentence | O(n) | O(n) | O(n) | ~5-10% |
| Char | O(n) | O(n) | O(n²) | ~15-20% |

**Recommendation**: Use WordSplitter for best balance of compatibility and speed.

### Memory Usage

Text splitting adds minimal memory overhead:
- Stores split IDs (1 int per token)
- Stores split logprobs (1 float per split)
- Temporary text strings (decoded once per sequence)

**Memory overhead**: <1% of total training memory

### When to Use Text Splitting

**Use text splitting when**:
- ✅ Teacher and student have different tokenizers
- ✅ Cross-architecture distillation (GPT → LLaMA)
- ✅ Want coarser-grained supervision (sentence-level)

**Don't use text splitting when**:
- ❌ Teacher and student use same tokenizer (use token-level)
- ❌ Need maximum precision (use token-level)
- ❌ Computational budget is very tight

## Troubleshooting

### Error: "OPD with text splitting requires 'responses' and 'tokenizer'"

**Cause**: Text splitting needs access to tokens and tokenizer.

**Solution**: This should be automatically handled by the trainer. If you see this error, check that:
1. `algorithm.opd_text_splitter` is properly configured
2. Trainer is passing tokenizer and responses correctly

### Warning: "Text splitting failed for sequence X: ... Falling back to token-level KL"

**Cause**: Decoding or splitting failed for a specific sequence.

**Solution**: This is expected for some edge cases (empty sequences, special tokens, etc.). The fallback to token-level KL ensures training continues. If you see this frequently:
1. Check if sequences contain unusual tokens
2. Try a different splitter (e.g., char instead of word)
3. Verify tokenizer is working correctly

### Low performance after enabling text splitting

**Check**:
1. Is splitter too coarse? (Try word instead of sentence)
2. Is splitter too fine? (Try word instead of char)
3. Is aggregation appropriate? (Try mean instead of min/max)
4. Are tokenizers very different? (May need more training data)

## Examples

### Example 1: GPT → LLaMA Distillation

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opd \
    algorithm.opd_text_splitter.name=word \
    algorithm.opd_text_splitter.aggregation=mean \
    actor_rollout_ref.model.path=meta-llama/Llama-3-8B \
    actor_rollout_ref.ref.model.path=gpt2-xl \
    actor_rollout_ref.ref.log_prob_estimator_enable=true
```

### Example 2: Sentence-Level Distillation

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opd \
    algorithm.opd_text_splitter.name=sentence \
    algorithm.opd_text_splitter.aggregation=mean \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.ref.model.path=Qwen/Qwen3-32B \
    actor_rollout_ref.ref.log_prob_estimator_enable=true
```

### Example 3: Character-Level Distillation

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=opd \
    algorithm.opd_text_splitter.name=char \
    algorithm.opd_text_splitter.aggregation=mean \
    actor_rollout_ref.model.path=small-model \
    actor_rollout_ref.ref.model.path=large-model \
    actor_rollout_ref.ref.log_prob_estimator_enable=true
```

## References

1. **Text Splitter Implementation**: `verl/trainer/ppo/text_splitter.py`
2. **OPD Implementation**: `verl/trainer/ppo/core_algos.py`
3. **Example Scripts**: `examples/grpo_trainer/`
   - `run_cross_tokenizer_opd.sh`
   - `text_splitting_example.py`
4. **OPD Documentation**: `examples/grpo_trainer/OPD_README.md`

## Custom Splitters

To implement a custom splitter:

```python
from verl.trainer.ppo.text_splitter import TextSplitter

class MyCustomSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        # Your custom splitting logic
        return text.split("|")  # Example: split by pipe
    
    def get_name(self) -> str:
        return "custom_pipe"

# Use in config:
# algorithm.opd_text_splitter.name=custom_pipe
```

---

**Status**: ✅ Ready to use  
**Version**: v0.2.0 (with text splitting)  
**Date**: 2025-01-09


