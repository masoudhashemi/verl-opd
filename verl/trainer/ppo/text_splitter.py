"""
Text splitting utilities for cross-tokenizer OPD (Online Policy Distillation).

When teacher and student models use different tokenizers, we need to:
1. Split text into coarser units (words, sentences, etc.)
2. Aggregate token-level logprobs to split-level logprobs
3. Compute KL divergence at the split level
4. Broadcast split-level advantages back to tokens

This enables distillation across different model architectures (e.g., GPT → LLaMA).
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import numpy as np
import torch


class TextSplitter(ABC):
    """
    Abstract base class for splitting text into coarser units.
    
    Subclasses implement different splitting strategies:
    - TokenSplitter: 1:1 mapping (when tokenizers match)
    - WordSplitter: Split by words
    - SentenceSplitter: Split by sentences
    - CharSplitter: Split by characters (fine-grained)
    - CustomSplitter: User-defined splitting logic
    """
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split text into segments.
        
        Args:
            text: Input text string
        
        Returns:
            List of text segments (e.g., words, sentences)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return a descriptive name for this splitter."""
        pass
    
    def map_tokens_to_splits(
        self,
        tokens: List[int],
        text: str,
        tokenizer: Any,
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Map token indices to split indices.
        
        Args:
            tokens: List of token IDs
            text: Original text (decoded from tokens)
            tokenizer: Tokenizer used to encode tokens
        
        Returns:
            Tuple of:
            - split_ids: List mapping each token to its split index
            - split_to_tokens: List of token indices for each split
        """
        splits = self.split_text(text)
        
        # Decode each token to text
        token_texts = []
        for token_id in tokens:
            try:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                token_texts.append(token_text)
            except Exception:
                token_texts.append("")
        
        # Map tokens to splits by matching text positions
        split_ids = []
        split_to_tokens = [[] for _ in range(len(splits))]
        
        current_split_idx = 0
        current_split_text = splits[current_split_idx] if splits else ""
        accumulated_text = ""
        
        for token_idx, token_text in enumerate(token_texts):
            accumulated_text += token_text
            
            # Check if we've completed the current split
            while current_split_idx < len(splits) and current_split_text in accumulated_text:
                # Move to next split
                accumulated_text = accumulated_text.replace(current_split_text, "", 1)
                current_split_idx += 1
                if current_split_idx < len(splits):
                    current_split_text = splits[current_split_idx]
            
            # Assign token to current split
            split_idx = min(current_split_idx, len(splits) - 1) if splits else 0
            split_ids.append(split_idx)
            split_to_tokens[split_idx].append(token_idx)
        
        return split_ids, split_to_tokens


class TokenSplitter(TextSplitter):
    """
    1:1 token-level splitting (default behavior when tokenizers match).
    
    Each token is its own split. This is the fastest and most precise option
    when student and teacher use the same tokenizer.
    """
    
    def split_text(self, text: str) -> List[str]:
        """Split into individual characters (approximation of tokens)."""
        return list(text)
    
    def get_name(self) -> str:
        return "token"
    
    def map_tokens_to_splits(
        self,
        tokens: List[int],
        text: str,
        tokenizer: Any,
    ) -> Tuple[List[int], List[List[int]]]:
        """Each token is its own split."""
        split_ids = list(range(len(tokens)))
        split_to_tokens = [[i] for i in range(len(tokens))]
        return split_ids, split_to_tokens


class WordSplitter(TextSplitter):
    """
    Split text by words (whitespace and punctuation).
    
    Good balance between granularity and cross-tokenizer compatibility.
    Works well when teacher and student have similar vocabularies.
    """
    
    def __init__(self, keep_whitespace: bool = True):
        """
        Args:
            keep_whitespace: If True, include whitespace in splits
        """
        self.keep_whitespace = keep_whitespace
    
    def split_text(self, text: str) -> List[str]:
        """Split by words using simple whitespace splitting."""
        import re
        
        if self.keep_whitespace:
            # Keep whitespace attached to words
            # Split on word boundaries but keep everything
            words = re.findall(r'\S+\s*|\s+', text)
        else:
            # Simple whitespace split
            words = text.split()
        
        return [w for w in words if w]  # Filter empty strings
    
    def get_name(self) -> str:
        return "word"


class SentenceSplitter(TextSplitter):
    """
    Split text by sentences.
    
    Provides coarse-grained supervision. Good for long-form generation
    where sentence-level coherence matters more than token-level accuracy.
    """
    
    def __init__(self, sentence_terminators: Optional[List[str]] = None):
        """
        Args:
            sentence_terminators: List of strings that end sentences
        """
        if sentence_terminators is None:
            sentence_terminators = ['.', '!', '?', '\n\n']
        self.sentence_terminators = sentence_terminators
    
    def split_text(self, text: str) -> List[str]:
        """Split by sentences using simple heuristics."""
        import re

        # Create regex pattern for sentence terminators
        pattern = '|'.join(re.escape(term) for term in self.sentence_terminators)
        
        # Split but keep the terminator
        sentences = re.split(f'({pattern})', text)
        
        # Combine terminator with preceding text
        result = []
        for i in range(0, len(sentences), 2):
            sent = sentences[i]
            if i + 1 < len(sentences):
                sent += sentences[i + 1]
            if sent.strip():
                result.append(sent)
        
        return result if result else [text]
    
    def get_name(self) -> str:
        return "sentence"


class FixedLengthSplitter(TextSplitter):
    """
    Split text into fixed-length chunks.
    
    Simple and predictable. Good for experimentation.
    """
    
    def __init__(self, chunk_size: int = 10):
        """
        Args:
            chunk_size: Number of characters per chunk
        """
        self.chunk_size = chunk_size
    
    def split_text(self, text: str) -> List[str]:
        """Split into fixed-length chunks."""
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
    
    def get_name(self) -> str:
        return f"fixed_{self.chunk_size}"


def aggregate_logprobs_by_splits(
    token_logprobs: torch.Tensor,
    split_ids: List[int],
    num_splits: int,
    aggregation: str = "mean",
) -> torch.Tensor:
    """
    Aggregate token-level log probabilities to split-level log probabilities.
    
    Args:
        token_logprobs: (num_tokens,) tensor of token log probs
        split_ids: List mapping each token to its split index
        num_splits: Total number of splits
        aggregation: How to aggregate logprobs within a split
            - "mean": Average log probs (default)
            - "sum": Sum log probs (product of probabilities)
            - "min": Min log prob (worst token in split)
            - "max": Max log prob (best token in split)
    
    Returns:
        (num_splits,) tensor of split-level log probs
    """
    split_logprobs = torch.zeros(num_splits, device=token_logprobs.device)
    split_counts = torch.zeros(num_splits, device=token_logprobs.device)
    
    # Aggregate logprobs for each split
    for token_idx, split_idx in enumerate(split_ids):
        if split_idx >= num_splits:
            continue
        
        if aggregation == "mean":
            split_logprobs[split_idx] += token_logprobs[token_idx]
            split_counts[split_idx] += 1
        elif aggregation == "sum":
            split_logprobs[split_idx] += token_logprobs[token_idx]
        elif aggregation == "min":
            if split_counts[split_idx] == 0:
                split_logprobs[split_idx] = token_logprobs[token_idx]
            else:
                split_logprobs[split_idx] = torch.min(
                    split_logprobs[split_idx], token_logprobs[token_idx]
                )
            split_counts[split_idx] = 1  # Mark as initialized
        elif aggregation == "max":
            if split_counts[split_idx] == 0:
                split_logprobs[split_idx] = token_logprobs[token_idx]
            else:
                split_logprobs[split_idx] = torch.max(
                    split_logprobs[split_idx], token_logprobs[token_idx]
                )
            split_counts[split_idx] = 1
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Normalize mean aggregation
    if aggregation == "mean":
        split_logprobs = split_logprobs / (split_counts + 1e-10)
    
    return split_logprobs


def broadcast_split_values_to_tokens(
    split_values: torch.Tensor,
    split_ids: List[int],
    num_tokens: int,
) -> torch.Tensor:
    """
    Broadcast split-level values back to token-level.
    
    Args:
        split_values: (num_splits,) tensor of values at split level
        split_ids: List mapping each token to its split index
        num_tokens: Total number of tokens
    
    Returns:
        (num_tokens,) tensor of token-level values
    """
    token_values = torch.zeros(num_tokens, device=split_values.device)
    
    for token_idx, split_idx in enumerate(split_ids):
        if split_idx < len(split_values):
            token_values[token_idx] = split_values[split_idx]
    
    return token_values


def get_text_splitter(splitter_name: str, **kwargs) -> TextSplitter:
    """
    Factory function to create text splitters.
    
    Args:
        splitter_name: Name of splitter ("token", "word", "sentence", "char", "fixed")
        **kwargs: Additional arguments for the splitter
    
    Returns:
        TextSplitter instance
    """
    splitter_registry = {
        "token": TokenSplitter,
        "word": WordSplitter,
        "sentence": SentenceSplitter,
        "fixed": FixedLengthSplitter,
    }
    
    if splitter_name not in splitter_registry:
        raise ValueError(
            f"Unknown splitter: {splitter_name}. "
            f"Available: {list(splitter_registry.keys())}"
        )
    
    return splitter_registry[splitter_name](**kwargs)


