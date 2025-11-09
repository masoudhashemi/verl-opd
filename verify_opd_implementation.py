#!/usr/bin/env python3
"""
Simple verification script for OPD implementation.
Run this to verify that all OPD files are present and syntactically correct.
"""

import os
import sys
from pathlib import Path


def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    exists = Path(filepath).exists()
    status = "✓" if exists else "✗"
    print(f"{status} {filepath}")
    return exists


def check_syntax(filepath: str) -> bool:
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        print(f"  ✓ Syntax OK")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax Error: {e}")
        return False


def main():
    print("="*80)
    print("OPD Implementation Verification")
    print("="*80)
    
    repo_root = Path(__file__).parent
    all_good = True
    
    print("\n1. Checking Modified Core Files:")
    print("-" * 80)
    
    core_files = [
        "verl/trainer/ppo/core_algos.py",
        "verl/trainer/ppo/ray_trainer.py",
    ]
    
    for filepath in core_files:
        full_path = repo_root / filepath
        exists = check_file_exists(str(full_path))
        if exists:
            syntax_ok = check_syntax(str(full_path))
            all_good = all_good and syntax_ok
        else:
            all_good = False
    
    print("\n2. Checking New Example Scripts:")
    print("-" * 80)
    
    example_files = [
        "examples/grpo_trainer/run_qwen3-8b_opd.sh",
        "examples/grpo_trainer/run_qwen3-32b_to_8b_opd.sh",
        "examples/grpo_trainer/opd_example.py",
    ]
    
    for filepath in example_files:
        full_path = repo_root / filepath
        exists = check_file_exists(str(full_path))
        if filepath.endswith('.py') and exists:
            syntax_ok = check_syntax(str(full_path))
            all_good = all_good and syntax_ok
        all_good = all_good and exists
    
    print("\n3. Checking Documentation:")
    print("-" * 80)
    
    doc_files = [
        "examples/grpo_trainer/OPD_README.md",
        "OPD_IMPLEMENTATION_SUMMARY.md",
    ]
    
    for filepath in doc_files:
        full_path = repo_root / filepath
        exists = check_file_exists(str(full_path))
        all_good = all_good and exists
    
    print("\n4. Checking Test Files:")
    print("-" * 80)
    
    test_files = [
        "tests/trainer/test_opd_advantage.py",
    ]
    
    for filepath in test_files:
        full_path = repo_root / filepath
        exists = check_file_exists(str(full_path))
        if exists:
            syntax_ok = check_syntax(str(full_path))
            all_good = all_good and syntax_ok
        all_good = all_good and exists
    
    print("\n5. Checking for Key Code Patterns:")
    print("-" * 80)
    
    # Check that OPD enum exists
    core_algos_path = repo_root / "verl/trainer/ppo/core_algos.py"
    if core_algos_path.exists():
        content = core_algos_path.read_text()
        
        patterns = [
            ('OPD = "opd"', "OPD enum in AdvantageEstimator"),
            ('compute_opd_advantage', "compute_opd_advantage function"),
            ('reverse_kl = student_log_probs - teacher_log_probs', "Reverse KL computation"),
        ]
        
        for pattern, description in patterns:
            found = pattern in content
            status = "✓" if found else "✗"
            print(f"{status} {description}")
            all_good = all_good and found
    
    # Check ray_trainer modifications
    ray_trainer_path = repo_root / "verl/trainer/ppo/ray_trainer.py"
    if ray_trainer_path.exists():
        content = ray_trainer_path.read_text()
        
        patterns = [
            ('compute_teacher_log_probs', "compute_teacher_log_probs function"),
            ('if adv_estimator == core_algos.AdvantageEstimator.OPD', "OPD advantage handling"),
            ('if self.config.algorithm.adv_estimator == "opd"', "OPD teacher logprob computation"),
        ]
        
        for pattern, description in patterns:
            found = pattern in content
            status = "✓" if found else "✗"
            print(f"{status} {description}")
            all_good = all_good and found
    
    print("\n" + "="*80)
    if all_good:
        print("✅ All checks passed! OPD implementation is complete.")
        print("\nNext steps:")
        print("1. Run unit tests: python -m pytest tests/trainer/test_opd_advantage.py")
        print("2. Try example script: bash examples/grpo_trainer/run_qwen3-8b_opd.sh")
        print("3. Read documentation: cat examples/grpo_trainer/OPD_README.md")
        return 0
    else:
        print("❌ Some checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

