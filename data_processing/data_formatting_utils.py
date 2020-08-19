import argparse
import difflib
import javalang
import json
import numpy as np
import os
import random
import re
import string

import sys
sys.path.append('../')
from diff_utils import INSERT, INSERT_END, REPLACE_OLD, REPLACE_NEW, REPLACE_END, DELETE, DELETE_END, KEEP, KEEP_END

SPECIAL_TAGS = ['{', '}', '@code', '@docRoot', '@inheritDoc', '@link', '@linkplain', '@value']

def subtokenize_comment(comment_line):
    """Subtokenizes a comment, which is in string format.
       Returns list of subtokens, labels (whether each term is a subtoken of a larger token),
       and indices (index of subtoken within larger token)."""
    comment_line = remove_return_string(comment_line)
    comment_line = remove_html_tag(comment_line.replace('/**', '').replace('**/', '').replace('/*', '').replace('*/', '').replace('*', '').strip())
    comment_line = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", comment_line.strip())
    comment_line = ' '.join(comment_line)
    comment_line = comment_line.replace('\n', ' ').strip()

    tokens = comment_line.split(' ')
    subtokens = []
    labels = []
    indices = []

    for token in tokens:
        curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
        try:
            new_curr = []
            for c in curr:
                by_symbol = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip())
                new_curr = new_curr + by_symbol

            curr = new_curr
        except:
            curr = []
        if len(curr) == 0:
            continue
        if len(curr) == 1:
            labels.append(0)
            indices.append(0)
            subtokens.append(curr[0].lower())
            continue
        
        for s, subtoken in enumerate(curr):
            labels.append(1)
            indices.append(s)
            subtokens.append(curr[s].lower())
    
    return subtokens, labels, indices

def subtokenize_code(line):
    """Subtokenizes a method, which is in string format.
       Returns list of subtokens, labels (whether each term is a subtoken of a larger token),
       and indices (index of subtoken within larger token)."""
    try:
        tokens = get_clean_code(list(javalang.tokenizer.tokenize(line)))
    except:
        tokens = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", line.strip())
    subtokens = []
    labels = []
    indices = []
    for token in tokens:
        curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
        if len(curr) == 0:
            continue
        if len(curr) == 1:
            labels.append(0)
            indices.append(0)
            subtokens.append(curr[0].lower())
            continue
        for s, subtoken in enumerate(curr):
            labels.append(1)
            indices.append(s)
            subtokens.append(curr[s].lower())
    
    return subtokens, labels, indices

def remove_html_tag(line):
    """Helper method for subtokenizing comment."""
    clean = re.compile('<.*?>')
    line = re.sub(clean, '', line)

    for tag in SPECIAL_TAGS:
        line = line.replace(tag, '')

    return line

def remove_return_string(line):
    """Helper method for subtokenizing comment."""
    return line.replace('@return', '').replace('@ return', '').strip()

def get_clean_code(tokenized_code):
    """Helper method for subtokenizing code."""
    token_vals = [t.value for t in tokenized_code]
    new_token_vals = []
    for t in token_vals:
        n = [c for c in re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", t.encode('ascii', errors='ignore').decode().strip()) if len(c) > 0]
        new_token_vals = new_token_vals + n

    token_vals = new_token_vals
    cleaned_code_tokens = []

    for c in token_vals:
        try:
            cleaned_code_tokens.append(str(c))
        except:
            pass

    return cleaned_code_tokens

def compute_code_diff_spans(old_tokens, old_labels, old_indices, new_tokens, new_labels, new_indices):
    spans = []
    labels = []
    indices = []

    for edit_type, o_start, o_end, n_start, n_end in difflib.SequenceMatcher(None, old_tokens, new_tokens).get_opcodes():
        if edit_type == 'equal':
            spans.extend([KEEP] + old_tokens[o_start:o_end] + [KEEP_END])
            labels.extend([0] + old_labels[o_start:o_end] + [0])
            indices.extend([0] + old_indices[o_start:o_end] + [0])
        elif edit_type == 'replace':
            spans.extend([REPLACE_OLD] + old_tokens[o_start:o_end] + [REPLACE_NEW] + new_tokens[n_start:n_end] + [REPLACE_END])
            labels.extend([0] + old_labels[o_start:o_end] + [0] + new_labels[n_start:n_end] + [0])
            indices.extend([0] + old_indices[o_start:o_end] + [0] + new_indices[n_start:n_end] + [0])
        elif edit_type == 'insert':
            spans.extend([INSERT] + new_tokens[n_start:n_end] + [INSERT_END])
            labels.extend([0] + new_labels[n_start:n_end] + [0])
            indices.extend([0] + new_indices[n_start:n_end] + [0])
        else:
            spans.extend([DELETE] + old_tokens[o_start:o_end] + [DELETE_END])
            labels.extend([0] + old_labels[o_start:o_end] + [0])
            indices.extend([0] + old_indices[o_start:o_end] + [0])

    return spans, labels, indices

