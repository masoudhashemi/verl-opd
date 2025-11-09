#!/usr/bin/env python3
"""
Example: Text Splitting for Cross-Tokenizer OPD

This script demonstrates how to use different text splitters to enable
distillation across models with different tokenizers.
"""

from verl.trainer.ppo.text_splitter import (CharSplitter, FixedLengthSplitter,
                                            SentenceSplitter, TokenSplitter,
                                            WordSplitter, get_text_splitter)


def demonstrate_text_splitters():
    """Show how different text splitters work."""
    
    # Example text
    text = "Hello world! This is a test. How are you doing today?"
    
    print("="*80)
    print("Text Splitting Examples")
    print("="*80)
    print(f"\nOriginal text: {text!r}\n")
    
    # 1. Word Splitter (most common for cross-tokenizer OPD)
    print("\n1. WORD SPLITTER (recommended for cross-tokenizer OPD)")
    print("-" * 80)
    word_splitter = WordSplitter(keep_whitespace=True)
    word_splits = word_splitter.split_text(text)
    print(f"Splitter: {word_splitter.get_name()}")
    print(f"Number of splits: {len(word_splits)}")
    print(f"Splits: {word_splits}")
    
    # 2. Sentence Splitter (for coarse-grained supervision)
    print("\n2. SENTENCE SPLITTER (coarse-grained)")
    print("-" * 80)
    sentence_splitter = SentenceSplitter()
    sentence_splits = sentence_splitter.split_text(text)
    print(f"Splitter: {sentence_splitter.get_name()}")
    print(f"Number of splits: {len(sentence_splits)}")
    print(f"Splits: {sentence_splits}")
    
    # 3. Character Splitter (fine-grained)
    print("\n3. CHARACTER SPLITTER (fine-grained)")
    print("-" * 80)
    char_splitter = CharSplitter()
    char_splits = char_splitter.split_text(text)
    print(f"Splitter: {char_splitter.get_name()}")
    print(f"Number of splits: {len(char_splits)} (showing first 20)")
    print(f"Splits: {char_splits[:20]}...")
    
    # 4. Fixed Length Splitter
    print("\n4. FIXED LENGTH SPLITTER")
    print("-" * 80)
    fixed_splitter = FixedLengthSplitter(chunk_size=10)
    fixed_splits = fixed_splitter.split_text(text)
    print(f"Splitter: {fixed_splitter.get_name()}")
    print(f"Number of splits: {len(fixed_splits)}")
    print(f"Splits: {fixed_splits}")
    
    # 5. Token Splitter (when tokenizers match)
    print("\n5. TOKEN SPLITTER (same tokenizer)")
    print("-" * 80)
    token_splitter = TokenSplitter()
    print(f"Splitter: {token_splitter.get_name()}")
    print(f"Note: This is the default when teacher and student use the same tokenizer")
    print(f"      It provides 1:1 token-level alignment (fastest, most precise)")
    
    # Using factory function
    print("\n6. USING FACTORY FUNCTION")
    print("-" * 80)
    splitter = get_text_splitter("word", keep_whitespace=True)
    splits = splitter.split_text(text)
    print(f"Created: {splitter.get_name()} splitter")
    print(f"Splits: {splits}")


def demonstrate_opd_config():
    """Show how to configure OPD with text splitting."""
    
    print("\n" + "="*80)
    print("OPD Configuration Examples")
    print("="*80)
    
    # Example 1: Word-level splitting (recommended)
    print("\n1. WORD-LEVEL SPLITTING (Recommended)")
    print("-" * 80)
    print("""
algorithm:
  adv_estimator: opd
  opd_text_splitter:
    name: word
    aggregation: mean  # mean, sum, min, max
    params:
      keep_whitespace: true
    """)
    
    # Example 2: Sentence-level splitting
    print("\n2. SENTENCE-LEVEL SPLITTING (Coarse-grained)")
    print("-" * 80)
    print("""
algorithm:
  adv_estimator: opd
  opd_text_splitter:
    name: sentence
    aggregation: mean
    params:
      sentence_terminators: ['.', '!', '?', '\\n\\n']
    """)
    
    # Example 3: Character-level splitting
    print("\n3. CHARACTER-LEVEL SPLITTING (Fine-grained)")
    print("-" * 80)
    print("""
algorithm:
  adv_estimator: opd
  opd_text_splitter:
    name: char
    aggregation: mean
    """)
    
    # Example 4: No splitting (same tokenizer)
    print("\n4. NO SPLITTING (Same tokenizer - default)")
    print("-" * 80)
    print("""
algorithm:
  adv_estimator: opd
  # No opd_text_splitter config = token-level KL
    """)
    
    print("\n" + "="*80)
    print("Command Line Examples")
    print("="*80)
    
    print("\n# Word-level OPD:")
    print("python -m verl.trainer.main_ppo \\")
    print("    algorithm.adv_estimator=opd \\")
    print("    algorithm.opd_text_splitter.name=word \\")
    print("    algorithm.opd_text_splitter.aggregation=mean \\")
    print("    ...")
    
    print("\n# Sentence-level OPD:")
    print("python -m verl.trainer.main_ppo \\")
    print("    algorithm.adv_estimator=opd \\")
    print("    algorithm.opd_text_splitter.name=sentence \\")
    print("    algorithm.opd_text_splitter.aggregation=mean \\")
    print("    ...")


def demonstrate_aggregation_modes():
    """Show different logprob aggregation modes."""
    
    print("\n" + "="*80)
    print("LogProb Aggregation Modes")
    print("="*80)
    
    print("""
When aggregating token-level logprobs to split-level logprobs, you can use:

1. MEAN (default, recommended):
   - Averages log probabilities within each split
   - Equivalent to geometric mean of probabilities
   - Most stable and commonly used
   
2. SUM:
   - Sums log probabilities within each split
   - Equivalent to product of probabilities
   - Emphasizes splits with more tokens
   
3. MIN:
   - Takes minimum (most negative) log prob in split
   - Focuses on worst token in each split
   - Useful for pessimistic distillation
   
4. MAX:
   - Takes maximum (least negative) log prob in split
   - Focuses on best token in each split
   - Useful for optimistic distillation

Example:
    algorithm.opd_text_splitter.aggregation=mean  # or sum, min, max
    """)


def main():
    """Run all examples."""
    demonstrate_text_splitters()
    demonstrate_opd_config()
    demonstrate_aggregation_modes()
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("""
TEXT SPLITTING FOR CROSS-TOKENIZER OPD:

✓ Use WordSplitter for most cross-tokenizer distillation tasks
✓ Use SentenceSplitter for coarse-grained, long-form generation
✓ Use CharSplitter for fine-grained control
✓ Use TokenSplitter (or no splitting) when tokenizers match

AGGREGATION:
✓ Use "mean" (default) for most stable results
✓ Use "sum" to weight splits by token count
✓ Use "min" for pessimistic distillation
✓ Use "max" for optimistic distillation

QUICK START:
✓ bash examples/grpo_trainer/run_cross_tokenizer_opd.sh

DOCUMENTATION:
✓ See verl/trainer/ppo/text_splitter.py for implementation
✓ See examples/grpo_trainer/OPD_README.md for full guide
    """)


if __name__ == "__main__":
    main()

