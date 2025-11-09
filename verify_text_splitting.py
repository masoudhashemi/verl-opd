#!/usr/bin/env python3
"""
Verification script for Text Splitting implementation.
Checks that all text splitting files are present and syntactically correct.
"""

import os
import sys
from pathlib import Path


def check_file(filepath: str, description: str) -> bool:
    """Check if a file exists and has valid syntax."""
    path = Path(filepath)
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    
    if exists and filepath.endswith('.py'):
        try:
            with open(filepath, 'r') as f:
                compile(f.read(), filepath, 'exec')
            print(f"  ✓ Syntax OK")
            return True
        except SyntaxError as e:
            print(f"  ✗ Syntax Error: {e}")
            return False
    return exists


def check_pattern(filepath: str, pattern: str, description: str) -> bool:
    """Check if a file contains a specific pattern."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            found = pattern in content
            status = "✓" if found else "✗"
            print(f"  {status} Contains: {description}")
            return found
    except Exception as e:
        print(f"  ✗ Error reading file: {e}")
        return False


def main():
    repo_root = Path(__file__).parent
    all_good = True
    
    print("="*80)
    print("Text Splitting Implementation Verification")
    print("="*80)
    
    print("\n1. Core Implementation Files")
    print("-" * 80)
    
    # Check text_splitter.py
    file_ok = check_file(
        str(repo_root / "verl/trainer/ppo/text_splitter.py"),
        "Text splitter module"
    )
    all_good = all_good and file_ok
    
    if file_ok:
        patterns = [
            ("class TextSplitter(ABC):", "TextSplitter base class"),
            ("class WordSplitter(TextSplitter):", "WordSplitter class"),
            ("class SentenceSplitter(TextSplitter):", "SentenceSplitter class"),
            ("class CharSplitter(TextSplitter):", "CharSplitter class"),
            ("def aggregate_logprobs_by_splits", "Aggregation function"),
            ("def broadcast_split_values_to_tokens", "Broadcasting function"),
            ("def get_text_splitter", "Factory function"),
        ]
        
        for pattern, desc in patterns:
            pattern_found = check_pattern(
                str(repo_root / "verl/trainer/ppo/text_splitter.py"),
                pattern,
                desc
            )
            all_good = all_good and pattern_found
    
    # Check core_algos.py modifications
    print()
    file_ok = check_file(
        str(repo_root / "verl/trainer/ppo/core_algos.py"),
        "Core algorithms module"
    )
    all_good = all_good and file_ok
    
    if file_ok:
        patterns = [
            ("use_text_splits = config is not None and config.get(\"opd_text_splitter\"", 
             "Text splitting enabled check"),
            ("from verl.trainer.ppo.text_splitter import", "Text splitter imports"),
            ("splitter = get_text_splitter", "Splitter initialization"),
            ("aggregate_logprobs_by_splits", "Logprob aggregation call"),
            ("broadcast_split_values_to_tokens", "Broadcasting call"),
        ]
        
        for pattern, desc in patterns:
            pattern_found = check_pattern(
                str(repo_root / "verl/trainer/ppo/core_algos.py"),
                pattern,
                desc
            )
            all_good = all_good and pattern_found
    
    # Check ray_trainer.py modifications
    print()
    file_ok = check_file(
        str(repo_root / "verl/trainer/ppo/ray_trainer.py"),
        "Ray trainer module"
    )
    all_good = all_good and file_ok
    
    if file_ok:
        patterns = [
            ("if config and config.get(\"opd_text_splitter\"", "Text splitter config check"),
            ("adv_kwargs[\"responses\"]", "Responses passing"),
            ("adv_kwargs[\"tokenizer\"]", "Tokenizer passing"),
        ]
        
        for pattern, desc in patterns:
            pattern_found = check_pattern(
                str(repo_root / "verl/trainer/ppo/ray_trainer.py"),
                pattern,
                desc
            )
            all_good = all_good and pattern_found
    
    print("\n2. Example Scripts")
    print("-" * 80)
    
    file_ok = check_file(
        str(repo_root / "examples/grpo_trainer/run_cross_tokenizer_opd.sh"),
        "Cross-tokenizer OPD script"
    )
    all_good = all_good and file_ok
    
    if file_ok:
        check_pattern(
            str(repo_root / "examples/grpo_trainer/run_cross_tokenizer_opd.sh"),
            "algorithm.opd_text_splitter.name=word",
            "Text splitter config"
        )
    
    print()
    file_ok = check_file(
        str(repo_root / "examples/grpo_trainer/text_splitting_example.py"),
        "Text splitting examples"
    )
    all_good = all_good and file_ok
    
    if file_ok:
        patterns = [
            ("from verl.trainer.ppo.text_splitter import", "Text splitter imports"),
            ("WordSplitter", "WordSplitter usage"),
            ("SentenceSplitter", "SentenceSplitter usage"),
            ("get_text_splitter", "Factory function usage"),
        ]
        
        for pattern, desc in patterns:
            check_pattern(
                str(repo_root / "examples/grpo_trainer/text_splitting_example.py"),
                pattern,
                desc
            )
    
    print("\n3. Documentation")
    print("-" * 80)
    
    file_ok = check_file(
        str(repo_root / "TEXT_SPLITTING_FOR_OPD.md"),
        "Text splitting documentation"
    )
    all_good = all_good and file_ok
    
    print()
    file_ok = check_file(
        str(repo_root / "OPD_TEXT_SPLITTING_SUMMARY.md"),
        "Implementation summary"
    )
    all_good = all_good and file_ok
    
    print("\n4. Summary")
    print("="*80)
    
    if all_good:
        print("✅ All checks passed! Text splitting is ready to use.")
        print("\nQuick Start:")
        print("  1. Run example: bash examples/grpo_trainer/run_cross_tokenizer_opd.sh")
        print("  2. Test splitters: python3 examples/grpo_trainer/text_splitting_example.py")
        print("  3. Read docs: cat TEXT_SPLITTING_FOR_OPD.md")
        print("\nConfiguration:")
        print("  algorithm:")
        print("    adv_estimator: opd")
        print("    opd_text_splitter:")
        print("      name: word  # word, sentence, char, fixed, token")
        print("      aggregation: mean  # mean, sum, min, max")
        return 0
    else:
        print("❌ Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


