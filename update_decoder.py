from dpu_utils.mlutils import Vocabulary
import numpy as np
import torch
from torch import nn

from constants import START, BEAM_SIZE
from decoder import Decoder


class UpdateDecoder(Decoder):
    def __init__(self, input_size, hidden_size, attention_state_size,
                 embedding_store, embedding_size, dropout_rate):
        """Decoder for the edit model which generates a sequence of NL edits based on learned representations of
           the old comment and code edits."""
        super(UpdateDecoder, self).__init__(input_size, hidden_size, attention_state_size,
            embedding_store, embedding_size, dropout_rate)
        
        self.attention_code_hidden_transform_matrix = nn.Parameter(
            torch.randn(self.attention_state_size, self.hidden_size,
                dtype=torch.float, requires_grad=True)
            )
        
        self.attention_old_nl_hidden_transform_matrix = nn.Parameter(
            torch.randn(self.attention_state_size, self.hidden_size,
                dtype=torch.float, requires_grad=True)
            )
        self.attention_output_layer = nn.Linear(2*self.attention_state_size + self.hidden_size,
            self.hidden_size, bias=False)

    def decode(self, initial_state, decoder_input_embeddings, encoder_hidden_states,
               code_hidden_states, old_nl_hidden_states, masks, code_masks, old_nl_masks):
        """Decoding with attention and copy. Attention is computed separately for each set of encoder hidden states."""
        decoder_states, decoder_final_state = self.gru.forward(decoder_input_embeddings,
            initial_state.unsqueeze(0))
        
        old_nl_alignment = torch.einsum('ijk,km,inm->inj', old_nl_hidden_states,
            self.attention_old_nl_hidden_transform_matrix, decoder_states)
        old_nl_alignment.masked_fill_(old_nl_masks, float('-inf'))
        old_nl_attention_scores = nn.functional.softmax(old_nl_alignment, dim=-1)
        old_nl_contexts = torch.einsum('ijk,ikm->ijm', old_nl_attention_scores, old_nl_hidden_states)
        
        code_alignment = torch.einsum('ijk,km,inm->inj', code_hidden_states,
            self.attention_code_hidden_transform_matrix, decoder_states)
        code_alignment.masked_fill_(code_masks, float('-inf'))
        code_attention_scores = nn.functional.softmax(code_alignment, dim=-1)
        code_contexts = torch.einsum('ijk,ikm->ijm', code_attention_scores, code_hidden_states)

        decoder_states = torch.tanh(self.attention_output_layer(
            torch.cat([old_nl_contexts, code_contexts, decoder_states], dim=-1)))
        
        generation_scores = torch.einsum('ijk,km->ijm', decoder_states, self.generation_output_matrix)
        copy_scores = torch.einsum('ijk,km,inm->inj', encoder_hidden_states,
            self.copy_encoder_hidden_transform_matrix, decoder_states)
        copy_scores.masked_fill_(masks, float('-inf'))

        combined_logprobs = nn.functional.log_softmax(torch.cat([generation_scores, copy_scores], dim=-1), dim=-1)
        generation_logprobs = combined_logprobs[:,:,:len(self.embedding_store.nl_vocabulary)]
        copy_logprobs = combined_logprobs[:, :,len(self.embedding_store.nl_vocabulary):]
        
        return decoder_states, decoder_final_state, generation_logprobs, copy_logprobs

    def forward(self, initial_state, decoder_input_embeddings, encoder_hidden_states,
                code_hidden_states, old_nl_hidden_states, masks, code_masks, old_nl_masks):
        """Runs decoding."""
        return self.decode(initial_state, decoder_input_embeddings, encoder_hidden_states,
            code_hidden_states, old_nl_hidden_states, masks, code_masks, old_nl_masks)
    
    def beam_decode(self, initial_state, encoder_hidden_states, code_hidden_states,
                    old_nl_hidden_states,masks, max_out_len, batch_data, code_masks, old_nl_masks, device):
        """Beam search. Generates the top K candidate predictions."""
        batch_size = initial_state.shape[0]
        decoded_batch = [list() for _ in range(batch_size)]
        decoded_batch_scores = np.zeros([batch_size, BEAM_SIZE])

        for b_idx in range(batch_size):
            beam_scores = torch.ones(BEAM_SIZE, dtype=torch.float32, device=device)
            beam_status = torch.zeros(BEAM_SIZE, dtype=torch.uint8, device=device)
            beam_predicted_ids = [list() for _ in range(BEAM_SIZE)]
            
            decoder_state = initial_state[b_idx].unsqueeze(0)
            decoder_input = torch.tensor([[self.embedding_store.get_nl_id(START)]], device=device)

            for i in range(max_out_len):
                beam_size = decoder_input.shape[0]
                tiled_encoder_states = encoder_hidden_states[b_idx].unsqueeze(0).expand(beam_size, -1, -1)
                tiled_masks = masks[b_idx].expand(beam_size, -1).unsqueeze(1)
                
                tiled_code_encoder_states = code_hidden_states[b_idx].unsqueeze(0).expand(beam_size, -1, -1)
                tiled_old_nl_encoder_states = old_nl_hidden_states[b_idx].unsqueeze(0).expand(beam_size, -1, -1)
                tiled_code_masks = code_masks[b_idx].expand(beam_size, -1).unsqueeze(1)
                tiled_old_nl_masks = old_nl_masks[b_idx].expand(beam_size, -1).unsqueeze(1)

                decoder_input_embeddings = self.embedding_store.get_nl_embeddings(decoder_input)
                decoder_attention_states, decoder_state, generation_logprobs, copy_logprobs = self.decode(decoder_state, decoder_input_embeddings,
                    tiled_encoder_states, tiled_code_encoder_states, tiled_old_nl_encoder_states, tiled_masks,
                    tiled_code_masks, tiled_old_nl_masks)
                
                generation_logprobs = generation_logprobs.squeeze(1)
                copy_logprobs = copy_logprobs.squeeze(1)

                prob_scores = torch.zeros([beam_size,
                    generation_logprobs.shape[-1] + copy_logprobs.shape[-1]], dtype=torch.float32, device=device)
                prob_scores[:, :generation_logprobs.shape[-1]] = torch.exp(generation_logprobs)
                for b in range(beam_size):
                    for c, inp_id in enumerate(batch_data.input_ids[b_idx]):
                        prob_scores[b, inp_id] = prob_scores[b, inp_id] + torch.exp(copy_logprobs[b,c])

                decoder_state = decoder_state.squeeze(0)
                top_scores_per_beam, top_indices_per_beam = torch.topk(prob_scores, k=BEAM_SIZE, dim=-1)
                top_scores_per_beam = top_scores_per_beam.reshape(-1)
                top_indices_per_beam = top_indices_per_beam.reshape(-1)

                full_scores = torch.zeros(beam_size * BEAM_SIZE, dtype=torch.float32, device=device)
                beam_positions = torch.zeros(beam_size * BEAM_SIZE, dtype=torch.int64, device=device)

                for beam_idx in range(beam_size):
                    if beam_status[beam_idx] == 1:
                        idx = beam_idx*beam_size
                        beam_positions[idx] = beam_idx
                        full_scores[idx] = beam_scores[beam_idx]
                        for sub_beam_idx in range(BEAM_SIZE):
                            idx = beam_idx*beam_size + sub_beam_idx
                            beam_positions[idx] = beam_idx
                        continue
                    else:
                        for sub_beam_idx in range(BEAM_SIZE):
                            idx = beam_idx*beam_size + sub_beam_idx
                            beam_positions[idx] = beam_idx
                            full_scores[idx] = beam_scores[beam_idx] * top_scores_per_beam[idx]
                
                # https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/9f6b66f43d2e05175dabcc024f79e1d37a667070/decode_beam.py#L124
                top_scores, top_indices = torch.topk(full_scores, k=BEAM_SIZE, dim=-1)
                new_scores = torch.ones(BEAM_SIZE, dtype=torch.float32, device=device)
                new_status = torch.zeros(BEAM_SIZE, dtype=torch.uint8, device=device)
                new_ids = [list() for _ in range(BEAM_SIZE)]
                next_step_ids = torch.zeros(BEAM_SIZE, dtype=torch.int64, device=device)
                next_decoder_state = torch.zeros([BEAM_SIZE, decoder_state.shape[1]], dtype=torch.float32, device=device)

                for b, pos in enumerate(top_indices):
                    beam_idx = beam_positions[pos]
                    next_decoder_state[b] = decoder_state[beam_idx]
                    if beam_status[beam_idx] == 1:
                        new_scores[b] = beam_scores[beam_idx]
                        new_status[b] = beam_status[beam_idx]
                        new_ids[b] = beam_predicted_ids[beam_idx]
                        next_step_ids[b] = self.embedding_store.get_end_id()
                    else:
                        new_scores[b] = top_scores[b]
                        predicted_id = top_indices_per_beam[pos]
                        new_status[b] = self.embedding_store.get_end_id() == predicted_id
                        new_ids[b] = beam_predicted_ids[beam_idx] + [predicted_id]
                        next_step_ids[b] = predicted_id
                
                unks = torch.ones(
                    next_step_ids.shape[0], dtype=torch.int64, device=device) * self.embedding_store.get_nl_id(Vocabulary.get_unk())
                decoder_input = torch.where(next_step_ids < len(self.embedding_store.nl_vocabulary), next_step_ids, unks).unsqueeze(1)
                decoder_state = next_decoder_state
                beam_scores = new_scores
                beam_status = new_status
                beam_predicted_ids = new_ids
        
            decoded_batch_scores[b_idx] = beam_scores
            decoded_batch[b_idx] = beam_predicted_ids

        return decoded_batch, decoded_batch_scores

 




