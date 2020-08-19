from abc import abstractmethod
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, attention_state_size, embedding_store,
                 embedding_size, dropout_rate):
        super(Decoder, self).__init__()
        self.input_size = input_size # Dimension of input into decoder cell
        self.hidden_size = hidden_size # Dimension of output from decoder cell
        self.attention_state_size = attention_state_size # Dimension of the encoder hidden states to attend to
        self.embedding_store = embedding_store
        self.gen_vocabulary_size = len(self.embedding_store.nl_vocabulary)
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True
        )

        # Parameters for attention
        self.attention_encoder_hidden_transform_matrix = nn.Parameter(
            torch.randn(self.attention_state_size, self.hidden_size,
                dtype=torch.float, requires_grad=True)
            )
        self.attention_output_layer = nn.Linear(self.attention_state_size + self.hidden_size,
            self.hidden_size, bias=False)
        
        # Parameters for generating/copying
        self.generation_output_matrix = nn.Parameter(
            torch.randn(self.hidden_size, self.gen_vocabulary_size,
                dtype=torch.float, requires_grad=True)
            )
        
        self.copy_encoder_hidden_transform_matrix = nn.Parameter(
            torch.randn(self.attention_state_size, self.hidden_size,
                dtype=torch.float, requires_grad=True)
            )
    
    @abstractmethod
    def decode(self):
        return NotImplemented
    
    @abstractmethod
    def forward(self, initial_state, decoder_input_embeddings, encoder_hidden_states, masks):
        return NotImplemented