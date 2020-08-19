import enum
from enum import Enum
import json
import torch
from typing import List, NamedTuple

class GenerationBatchData(NamedTuple):
    """Stores tensorized batch used in generation model."""
    code_ids: torch.Tensor
    code_lengths: torch.Tensor
    trg_nl_ids: torch.Tensor
    trg_extended_nl_ids: torch.Tensor
    trg_nl_lengths: torch.Tensor
    invalid_copy_positions: torch.Tensor
    input_str_reps: List[List[str]]
    input_ids: List[List[str]]

class UpdateBatchData(NamedTuple):
    """Stores tensorized batch used in edit model."""
    code_ids: torch.Tensor
    code_lengths: torch.Tensor
    old_nl_ids: torch.Tensor
    old_nl_lengths: torch.Tensor
    trg_nl_ids: torch.Tensor
    trg_extended_nl_ids: torch.Tensor
    trg_nl_lengths: torch.Tensor
    invalid_copy_positions: torch.Tensor
    input_str_reps: List[List[str]]
    input_ids: List[List[str]]
    code_features: torch.Tensor
    nl_features: torch.Tensor

class Example(NamedTuple):
    """Data format for examples used in generation model."""
    id: str
    old_comment: str
    old_comment_tokens: List[str]
    new_comment: str
    new_comment_tokens: List[str]
    old_code: str
    old_code_tokens: List[str]
    new_code: str
    new_code_tokens: List[str]

class DiffExample(NamedTuple):
    """Data format for examples used in edit model."""
    id: str
    old_comment: str
    old_comment_tokens: List[str]
    new_comment: str
    new_comment_tokens: List[str]
    old_code: str
    old_code_tokens: List[str]
    new_code: str
    new_code_tokens: List[str]
    span_diff_code: str
    span_diff_code_tokens: List[str]
    span_minimal_diff_comment: str
    span_minimal_diff_comment_tokens: List[str]
    token_diff_code_tokens: List[str]

def read_examples_from_file(filename):
    """Reads in data in the format used for edit model."""
    with open(filename) as f:
        data = json.load(f)
    return [DiffExample(**d) for d in data]

def read_full_examples_from_file(filename):
    """Reads in data in the format used for generation model."""
    with open(filename) as f:
        data = json.load(f)
    return [Example(**d) for d in data]