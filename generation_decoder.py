from dpu_utils.mlutils import Vocabulary
import logging
import numpy as np
import os
import random
import sys
import torch
from torch import nn

from constants import START, BEAM_SIZE
from decoder import Decoder

class GenerationDecoder(Decoder):
    def __init__(self, input_size, hidden_size, attention_state_size, embedding_store,
                 embedding_size, dropout_rate):
        """Decoder for the generation model which generates a comment based on a
           learned representation of a method."""
        super(GenerationDecoder, self).__init__(input_size, hidden_size, attention_state_size,
            embedding_store, embedding_size, dropout_rate)
    
    def decode(self, initial_state, decoder_input_embeddings, encoder_hidden_states, masks):
        """Decoding with attention and copy."""
        decoder_states, decoder_final_state = self.gru.forward(decoder_input_embeddings,
            initial_state.unsqueeze(0))

        # https://stackoverflow.com/questions/50571991/implementing-luong-attention-in-pytorch
        attn_alignment = torch.einsum('ijk,km,inm->inj', encoder_hidden_states,
            self.attention_encoder_hidden_transform_matrix, decoder_states)
        attn_alignment.masked_fill_(masks, float('-inf'))
        attention_scores = nn.functional.softmax(attn_alignment, dim=-1)
        contexts = torch.einsum('ijk,ikm->ijm', attention_scores, encoder_hidden_states)
        decoder_states = torch.tanh(self.attention_output_layer(torch.cat([contexts, decoder_states], dim=-1)))
        
        generation_scores = torch.einsum('ijk,km->ijm', decoder_states, self.generation_output_matrix)
        copy_scores = torch.einsum('ijk,km,inm->inj', encoder_hidden_states,
            self.copy_encoder_hidden_transform_matrix, decoder_states)
        copy_scores.masked_fill_(masks, float('-inf'))

        combined_logprobs = nn.functional.log_softmax(torch.cat([generation_scores, copy_scores], dim=-1), dim=-1)
        generation_logprobs = combined_logprobs[:,:,:len(self.embedding_store.nl_vocabulary)]
        copy_logprobs = combined_logprobs[:, :,len(self.embedding_store.nl_vocabulary):]
        
        return decoder_states, decoder_final_state, generation_logprobs, copy_logprobs

    def forward(self, initial_state, decoder_input_embeddings, encoder_hidden_states, masks):
        """Runs decoding."""
        return self.decode(initial_state, decoder_input_embeddings, encoder_hidden_states, masks)

    def greedy_decode(self, initial_state, encoder_hidden_states, masks, max_out_len, batch_data, device):
        """Greedily generates the output sequence."""
        # Derived from https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/9f6b66f43d2e05175dabcc024f79e1d37a667070/decode_beam.py#L163
        batch_size = initial_state.shape[0]
        decoder_state = initial_state
        decoder_input = torch.tensor(
            [[self.embedding_store.get_nl_id(START)]] * batch_size,
            device=device
        )

        decoded_batch = np.zeros([batch_size, max_out_len], dtype=np.int64)
        decoded_batch_scores = np.zeros([batch_size, max_out_len])
        
        for i in range(max_out_len):
            decoder_input_embeddings = self.embedding_store.get_nl_embeddings(decoder_input)
            decoder_attention_states, decoder_state, generation_logprobs, copy_logprobs = self.decode(decoder_state,
                decoder_input_embeddings, encoder_hidden_states, masks)
            
            generation_logprobs = generation_logprobs.squeeze(1)
            copy_logprobs = copy_logprobs.squeeze(1)
            
            prob_scores = torch.zeros([generation_logprobs.shape[0],
                generation_logprobs.shape[-1] + copy_logprobs.shape[-1]], dtype=torch.float32, device=device)
            prob_scores[:, :generation_logprobs.shape[-1]] = torch.exp(generation_logprobs)
            for b in range(generation_logprobs.shape[0]):
                for c, inp_id in enumerate(batch_data.input_ids[b]):
                    prob_scores[b, inp_id] = prob_scores[b, inp_id] + torch.exp(copy_logprobs[b,c])
            
            predicted_ids = torch.argmax(prob_scores, dim=-1)
            decoded_batch_scores[:, i] = prob_scores[torch.arange(prob_scores.shape[0]), predicted_ids]
            decoded_batch[:, i] = predicted_ids
            
            unks = torch.ones(
                predicted_ids.shape[0], dtype=torch.int64, device=device) * self.embedding_store.get_nl_id(Vocabulary.get_unk())
            decoder_input = torch.where(predicted_ids < len(self.embedding_store.nl_vocabulary), predicted_ids, unks).unsqueeze(1)
            decoder_state = decoder_state.squeeze(0)
            
        return decoded_batch, decoded_batch_scores
 




