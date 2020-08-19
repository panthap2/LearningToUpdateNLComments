import numpy as np
import torch

def merge_encoder_outputs(a_states, a_lengths, b_states, b_lengths, device):
    a_max_len = a_states.size(1)
    b_max_len = b_states.size(1)
    combined_len = a_max_len + b_max_len
    padded_b_states = torch.zeros([b_states.size(0), combined_len, b_states.size(-1)], device=device)
    padded_b_states[:, :b_max_len, :] = b_states    
    full_matrix = torch.cat([a_states, padded_b_states], dim=1)
    a_idxs = torch.arange(combined_len, dtype=torch.long, device=device).view(-1, 1)
    b_idxs = torch.arange(combined_len, dtype=torch.long,
                    device=device).view(-1,1) - a_lengths.view(1, -1) + a_max_len
    idxs = torch.where(b_idxs < a_max_len, a_idxs, b_idxs).permute(1, 0)
    offset = torch.arange(0, full_matrix.size(0) * full_matrix.size(1), full_matrix.size(1), device=device)
    idxs = idxs + offset.unsqueeze(1)
    combined_states = full_matrix.reshape(-1, full_matrix.shape[-1])[idxs]
    combined_lengths = a_lengths + b_lengths

    return combined_states, combined_lengths

def get_invalid_copy_locations(input_sequence, max_input_length, output_sequence, max_output_length):
    input_length = min(len(input_sequence), max_input_length)
    output_length = min(len(output_sequence), max_output_length)

    invalid_copy_locations = np.ones([max_output_length, max_input_length], dtype=np.bool)
    for o in range(output_length):
        for i in range(input_length):
            invalid_copy_locations[o,i] = output_sequence[o] != input_sequence[i]

    return invalid_copy_locations