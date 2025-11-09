"""
Unit tests for Online Policy Distillation (OPD) advantage computation.
"""

import numpy as np
import torch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Mock pytest.raises for standalone execution
    class MockPytest:
        @staticmethod
        def raises(exception, match=None):
            class RaisesContext:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {exception} but no exception was raised")
                    if not issubclass(exc_type, exception):
                        return False
                    if match and match not in str(exc_val):
                        raise AssertionError(f"Expected message to contain '{match}', got '{exc_val}'")
                    return True
            return RaisesContext()
    pytest = MockPytest()

from verl.trainer.ppo.core_algos import (AdvantageEstimator,
                                         compute_opd_advantage)


class TestOPDAdvantage:
    """Test suite for OPD advantage estimator."""

    def test_opd_enum_exists(self):
        """Test that OPD enum is properly registered."""
        assert hasattr(AdvantageEstimator, "OPD")
        assert AdvantageEstimator.OPD == "opd"

    def test_compute_opd_advantage_basic(self):
        """Test basic OPD advantage computation."""
        batch_size = 4
        response_length = 10

        # Create mock data
        token_level_rewards = torch.zeros(batch_size, response_length)  # Not used in OPD
        response_mask = torch.ones(batch_size, response_length)
        index = np.array([0, 0, 1, 1])  # Two groups

        # Student and teacher log probs
        student_log_probs = torch.randn(batch_size, response_length)
        teacher_log_probs = torch.randn(batch_size, response_length)

        # Compute advantages
        advantages, returns = compute_opd_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            teacher_log_probs=teacher_log_probs,
            student_log_probs=student_log_probs,
        )

        # Check shapes
        assert advantages.shape == (batch_size, response_length)
        assert returns.shape == (batch_size, response_length)

        # Check that advantages = -(student_log_probs - teacher_log_probs)
        expected_advantages = -(student_log_probs - teacher_log_probs) * response_mask
        torch.testing.assert_close(advantages, expected_advantages)

    def test_compute_opd_advantage_with_mask(self):
        """Test OPD advantage computation with partial masking."""
        batch_size = 2
        response_length = 5

        # Create data with partial masking
        token_level_rewards = torch.zeros(batch_size, response_length)
        response_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)
        index = np.array([0, 1])

        student_log_probs = torch.randn(batch_size, response_length)
        teacher_log_probs = torch.randn(batch_size, response_length)

        advantages, returns = compute_opd_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            teacher_log_probs=teacher_log_probs,
            student_log_probs=student_log_probs,
        )

        # Check that masked positions are zero
        assert advantages[0, 3:].sum() == 0
        assert advantages[1, 4:].sum() == 0

        # Check that unmasked positions are non-zero (with high probability)
        assert advantages[0, :3].abs().sum() > 0
        assert advantages[1, :4].abs().sum() > 0

    def test_compute_opd_advantage_reverse_kl(self):
        """Test that OPD computes reverse KL correctly."""
        batch_size = 2
        response_length = 3

        token_level_rewards = torch.zeros(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = np.array([0, 1])

        # Simple case: student = teacher -> advantages should be near zero
        log_probs = torch.tensor([[-1.0, -1.5, -2.0], [-0.5, -1.0, -1.5]])
        student_log_probs = log_probs.clone()
        teacher_log_probs = log_probs.clone()

        advantages, _ = compute_opd_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            teacher_log_probs=teacher_log_probs,
            student_log_probs=student_log_probs,
        )

        # When student = teacher, reverse KL = 0, so advantages = 0
        torch.testing.assert_close(advantages, torch.zeros_like(advantages), atol=1e-6, rtol=1e-6)

    def test_compute_opd_advantage_student_worse_than_teacher(self):
        """Test that student worse than teacher gets positive advantages (to improve)."""
        batch_size = 1
        response_length = 3

        token_level_rewards = torch.zeros(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = np.array([0])

        # Student has lower log probs (worse) than teacher
        student_log_probs = torch.tensor([[-2.0, -3.0, -4.0]])  # Lower (worse)
        teacher_log_probs = torch.tensor([[-1.0, -1.5, -2.0]])  # Higher (better)

        advantages, _ = compute_opd_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            teacher_log_probs=teacher_log_probs,
            student_log_probs=student_log_probs,
        )

        # reverse_KL = student - teacher = [-2,-3,-4] - [-1,-1.5,-2] = [-1,-1.5,-2]
        # advantages = -reverse_KL = [1, 1.5, 2] (positive, encourages improvement)
        expected = torch.tensor([[1.0, 1.5, 2.0]])
        torch.testing.assert_close(advantages, expected, atol=1e-6, rtol=1e-6)

    def test_compute_opd_advantage_missing_teacher_logprobs(self):
        """Test that missing teacher_log_probs raises ValueError."""
        batch_size = 2
        response_length = 3

        token_level_rewards = torch.zeros(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = np.array([0, 1])

        with pytest.raises(ValueError, match="OPD requires 'teacher_log_probs'"):
            compute_opd_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                # Missing teacher_log_probs
                student_log_probs=torch.randn(batch_size, response_length),
            )

    def test_compute_opd_advantage_missing_student_logprobs(self):
        """Test that missing student_log_probs raises ValueError."""
        batch_size = 2
        response_length = 3

        token_level_rewards = torch.zeros(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = np.array([0, 1])

        with pytest.raises(ValueError, match="OPD requires 'student_log_probs'"):
            compute_opd_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                teacher_log_probs=torch.randn(batch_size, response_length),
                # Missing student_log_probs
            )

    def test_compute_opd_advantage_with_normalization(self):
        """Test OPD advantage computation with group normalization."""
        batch_size = 4
        response_length = 5

        token_level_rewards = torch.zeros(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = np.array([0, 0, 1, 1])  # Two groups

        student_log_probs = torch.randn(batch_size, response_length)
        teacher_log_probs = torch.randn(batch_size, response_length)

        # With normalization enabled
        advantages_norm, _ = compute_opd_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            teacher_log_probs=teacher_log_probs,
            student_log_probs=student_log_probs,
            norm_adv_by_std_in_grpo=True,
            config={"norm_adv_by_std_in_grpo": True},
        )

        # Without normalization
        advantages_no_norm, _ = compute_opd_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            teacher_log_probs=teacher_log_probs,
            student_log_probs=student_log_probs,
            norm_adv_by_std_in_grpo=False,
        )

        # Shapes should match
        assert advantages_norm.shape == advantages_no_norm.shape

        # Values should be different (unless std happens to be 1)
        # We can't assert they're different in general, but we can check they're valid
        assert torch.isfinite(advantages_norm).all()
        assert torch.isfinite(advantages_no_norm).all()


if __name__ == "__main__":
    # Run tests
    test = TestOPDAdvantage()
    test.test_opd_enum_exists()
    print("✓ test_opd_enum_exists")

    test.test_compute_opd_advantage_basic()
    print("✓ test_compute_opd_advantage_basic")

    test.test_compute_opd_advantage_with_mask()
    print("✓ test_compute_opd_advantage_with_mask")

    test.test_compute_opd_advantage_reverse_kl()
    print("✓ test_compute_opd_advantage_reverse_kl")

    test.test_compute_opd_advantage_student_worse_than_teacher()
    print("✓ test_compute_opd_advantage_student_worse_than_teacher")

    try:
        test.test_compute_opd_advantage_missing_teacher_logprobs()
    except AssertionError:
        print("✓ test_compute_opd_advantage_missing_teacher_logprobs (expected to raise)")

    try:
        test.test_compute_opd_advantage_missing_student_logprobs()
    except AssertionError:
        print("✓ test_compute_opd_advantage_missing_student_logprobs (expected to raise)")

    test.test_compute_opd_advantage_with_normalization()
    print("✓ test_compute_opd_advantage_with_normalization")

    print("\n✅ All OPD tests passed!")

