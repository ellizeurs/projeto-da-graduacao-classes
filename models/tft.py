import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .torch_generic_model import TorchGenericModel

from .window.sliding_window import SlidingWindow


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, context_size):
        super(VariableSelectionNetwork, self).__init__()
        self.context_projection = nn.Linear(context_size, input_size)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, x, context):
        context = self.context_projection(context)
        attention_weights = self.softmax(context)
        selected_input = x * attention_weights.unsqueeze(-1)
        return self.linear(selected_input)


class TFT(TorchGenericModel):
    def __init__(
        self,
        input_chunk_length,
        output_chunk_length,
        hidden_size=128,
        lstm_layers=1,
        dropout=0.1,
        num_attention_heads=4,
        batch_size=32,
        lr=1e-3,
        n_epochs=100,
        random_state=None,
        loss_fn=torch.nn.MSELoss(),
        optimizer_cls=torch.optim.Adam,
        window_model=SlidingWindow(),
        preprocessing=None,
        pl_trainer_kwargs=None,
    ):

        super().__init__(
            input_chunk_length,
            output_chunk_length,
            batch_size,
            loss_fn,
            n_epochs,
            random_state,
            window_model,
            preprocessing,
            pl_trainer_kwargs,
        )

        self.hidden_size = hidden_size

        # LSTM layers
        self.lstm_encoder = nn.LSTM(
            input_chunk_length,
            hidden_size,
            lstm_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.lstm_decoder = nn.LSTM(
            output_chunk_length,
            hidden_size,
            lstm_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Transformer layers
        self.encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_attention_heads, dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=1)

        # Variable selection network
        self.variable_selection = VariableSelectionNetwork(
            input_chunk_length, hidden_size, context_size=hidden_size
        )

        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_size, output_chunk_length)

        self.optimizer = optimizer_cls(self.parameters(), lr=lr)

    def forward(self, x):
        # LSTM encoding
        encoder_output, (h_n, c_n) = self.lstm_encoder(x)

        # Transformer encoding
        transformer_output = self.transformer_encoder(encoder_output)

        # Variable selection
        selected_output = self.variable_selection(
            transformer_output, context=transformer_output
        )

        # LSTM decoding
        decoder_output, _ = self.lstm_decoder(selected_output, (h_n, c_n))

        # Final output
        output = self.fc(decoder_output)

        return output
