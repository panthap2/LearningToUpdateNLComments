import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout):
        """Bi-directional GRU encoder which learns a representation for an input sequence."""
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size=embedding_size,
                        hidden_size=hidden_size,
                        dropout=dropout,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=True)
    
    def forward(self, src_embedded_tokens, src_lengths, device):
        """Encodes input sequence and returns corresponding hidden states and final state."""
        encoder_hidden_states, _ = self.rnn.forward(src_embedded_tokens)
        encoder_final_state = encoder_hidden_states[torch.arange(
            src_embedded_tokens.size()[0], dtype=torch.int64, device=device), src_lengths-1]
        return encoder_hidden_states, encoder_final_state