from data_formatting_utils import subtokenize_comment, subtokenize_code, compute_code_diff_spans
from data_utils import DiffExample
from method_details_extraction import extract_method_name, extract_return_type, extract_return_statements

import sys
sys.path.append('../')
from diff_utils import compute_minimal_comment_diffs, compute_code_diffs

old_nl = '@return the highest value from examScores list'
old_code = 'public SpecialValue getBestValue()\n{\n\treturn Collections.max(values);\n}'

new_nl = '@return the lowest value from examScores list'
new_code = 'public SpecialValue getBestValue()\n{\n\treturn Collections.min(values);\n}'

old_nl_subtokens, old_nl_subtoken_labels, old_nl_subtoken_indices = subtokenize_comment(old_nl)
old_code_subtokens, old_code_subtoken_labels, old_code_subtoken_indices = subtokenize_code(old_code)

new_nl_subtokens, new_nl_subtoken_labels, new_nl_subtoken_indices = subtokenize_comment(new_nl)
new_code_subtokens, new_code_subtoken_labels, new_code_subtoken_indices = subtokenize_code(new_code)

span_diff_tokens, span_diff_labels, span_diff_indices = compute_code_diff_spans(
    old_code_subtokens, old_code_subtoken_labels, old_code_subtoken_indices, new_code_subtokens, new_code_subtoken_labels, new_code_subtoken_indices)

_, token_diff_tokens, _ = compute_code_diffs(old_code_subtokens, new_code_subtokens)

comment_edit_spans, _, _ = compute_minimal_comment_diffs(old_nl_subtokens, new_nl_subtokens)

example = DiffExample(
    id = 'test_id',
    old_comment = ' '.join(old_nl_subtokens),
    old_comment_tokens = old_nl_subtokens,
    new_comment = ' '.join(new_nl_subtokens),
    new_comment_tokens = new_nl_subtokens,
    old_code = ' '.join(old_code_subtokens),
    old_code_tokens = old_code_subtokens,
    new_code = ' '.join(new_code_subtokens),
    new_code_tokens = new_code_subtokens,
    span_diff_code = ' '.join(span_diff_tokens),
    span_diff_code_tokens = span_diff_tokens,
    span_minimal_diff_comment = ' '.join(comment_edit_spans),
    span_minimal_diff_comment_tokens = comment_edit_spans,
    token_diff_code_tokens = token_diff_tokens
)

# Should be stored in 'resources/method_details.json' for every example, by id
method_details = dict()
method_details[example.id] = dict()
method_details[example.id]['method_name_subtokens'], _, _ = subtokenize_code(extract_method_name(old_code.split('\n')))
method_details[example.id]['old_return_type_subtokens'], _, _ = subtokenize_code(extract_return_type(old_code.split('\n')))
method_details[example.id]['return_type_subtokens'], _, _ = subtokenize_code(extract_return_type(new_code.split('\n')))
method_details[example.id]['old_return_sequence'], _, _ = subtokenize_code(' '.join(extract_return_statements(old_code.split('\n'))))
method_details[example.id]['new_return_sequence'], _, _ = subtokenize_code(' '.join(extract_return_statements(new_code.split('\n'))))
method_details[example.id]['old_code'] = old_code
method_details[example.id]['new_code'] = new_code

# Should be stored in 'resources/tokenization_features.json' for every example, by id
tokenization_features = dict()
tokenization_features[example.id] = dict()
tokenization_features[example.id]['edit_span_subtoken_labels'] = span_diff_labels
tokenization_features[example.id]['edit_span_subtoken_indices'] = span_diff_indices
tokenization_features[example.id]['old_nl_subtoken_labels'] = old_nl_subtoken_labels
tokenization_features[example.id]['old_nl_subtoken_indices'] = old_nl_subtoken_indices