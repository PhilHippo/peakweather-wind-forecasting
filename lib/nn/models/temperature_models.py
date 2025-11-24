import torch
import torch.nn as nn
from einops import rearrange
from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, DiffConv
from tsl.nn.models import BaseModel

# MODEL 0: TCN
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=self.padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=self.padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)[:, :, :-self.padding]
        out = self.dropout1(self.relu1(self.bn1(out)))
        out = self.conv2(out)[:, :, :-self.padding]
        out = self.dropout2(self.relu2(self.bn2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(BaseModel):
    def __init__(self, input_size, n_nodes, horizon, exog_size=0, output_size=None, hidden_size=32, num_layers=3, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        self.input_proj = nn.Linear(input_size + exog_size, hidden_size)
        layers = [TemporalBlock(hidden_size, hidden_size, kernel_size, 2**i, dropout) for i in range(num_layers)]
        self.tcn = nn.Sequential(*layers)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon * (output_size if output_size is not None else 1))
        )
        self.output_size = output_size
        self.horizon = horizon
        
    def forward(self, x, u=None, edge_index=None, edge_weight=None, mc_samples=None):
        if u is not None:
            if u.shape[1] > x.shape[1]: u = u[:, :x.shape[1]]
            x = torch.cat([x, u], dim=-1)
        b, t, n, f = x.shape
        x = rearrange(x, 'b t n f -> (b n) t f')
        x = self.input_proj(x)
        x = rearrange(x, '(bn) t h -> (bn) h t', bn=b*n)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.output_proj(x)
        x = rearrange(x, '(b n) (t f) -> b t n f', b=b, n=n, t=self.horizon, f=self.output_size if self.output_size is not None else 1)
        
        # Generate samples if mc_samples is provided
        if mc_samples is not None:
            # Add Gaussian noise for sampling (similar to PersistenceModel)
            sigma = x.std(dim=(1, 2), keepdim=True)  # Compute std across time and nodes
            noise_shape = (mc_samples, x.size(0), self.horizon, x.size(2), x.size(3))
            x = x.unsqueeze(0) + sigma * torch.randn(*noise_shape, device=x.device)
            # x now has shape [mc_samples, batch, horizon, nodes, features]
        
        return x


# MODEL 1: Enhanced RNN
class EnhancedRNNModel(BaseModel):
    def __init__(self, input_size, n_nodes, horizon, exog_size=0, output_size=None, hidden_size=64, emb_size=32, rnn_layers=2, dropout=0.2):
        super(EnhancedRNNModel, self).__init__()
        self.encoder = nn.Linear(input_size + exog_size, hidden_size)
        self.node_embeddings = NodeEmbedding(n_nodes, emb_size)
        self.emb_proj = nn.Linear(emb_size, hidden_size)
        self.rnn = RNN(input_size=hidden_size, hidden_size=hidden_size, n_layers=rnn_layers, cell='gru', return_only_last_state=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, dropout=dropout, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon * (output_size if output_size is not None else 1))
        )
        self.output_size = output_size
        self.horizon = horizon
        
    def forward(self, x, u=None, edge_index=None, edge_weight=None, mc_samples=None):
        if u is not None:
            if u.shape[1] > x.shape[1]: u = u[:, :x.shape[1]]
            x = torch.cat([x, u], dim=-1)
        b, t, n, f = x.shape
        x_enc = self.encoder(x)
        emb = self.node_embeddings(expand=(b, -1, -1))
        x_enc = x_enc + self.emb_proj(emb).unsqueeze(1)
        h = self.rnn(x_enc)
        h_att, _ = self.attention(h, h, h)
        h = h + h_att
        out = self.decoder(h)
        out = rearrange(out, 'b n (t f) -> b t n f', t=self.horizon, f=self.output_size if self.output_size is not None else 1)
        
        # Generate samples if mc_samples is provided
        if mc_samples is not None:
            sigma = out.std(dim=(1, 2), keepdim=True)
            noise_shape = (mc_samples, out.size(0), self.horizon, out.size(2), out.size(3))
            out = out.unsqueeze(0) + sigma * torch.randn(*noise_shape, device=out.device)
        
        return out

# MODEL 2: Improved STGNN
class ImprovedSTGNN(BaseModel):
    def __init__(self, input_size, n_nodes, horizon, exog_size=0, output_size=None, hidden_size=64, emb_size=32, rnn_layers=2, gnn_layers=2, gnn_kernel=2, dropout=0.2):
        super(ImprovedSTGNN, self).__init__()
        self.encoder = nn.Linear(input_size + exog_size, hidden_size)
        self.node_embeddings = NodeEmbedding(n_nodes, emb_size)
        self.temporal_encoder = RNN(input_size=hidden_size + emb_size, hidden_size=hidden_size, n_layers=rnn_layers, cell='gru', return_only_last_state=True, dropout=dropout)
        self.spatial_layers = nn.ModuleList([
            DiffConv(in_channels=hidden_size + emb_size if i == 0 else hidden_size, out_channels=hidden_size, k=gnn_kernel)
            for i in range(gnn_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(gnn_layers)])
        self.skip_proj = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon * (output_size if output_size is not None else 1))
        )
        self.output_size = output_size
        self.horizon = horizon
        
    def forward(self, x, edge_index, edge_weight, u=None, mc_samples=None):
        if u is not None:
            if u.shape[1] > x.shape[1]: u = u[:, :x.shape[1]]
            x = torch.cat([x, u], dim=-1)
        b, t, n, f = x.shape
        x_enc = self.encoder(x)
        emb = self.node_embeddings(expand=(b, -1, -1))
        x_emb = torch.cat([x_enc, emb.unsqueeze(1).expand(-1, t, -1, -1)], dim=-1)
        h_temp = self.temporal_encoder(x_emb)
        h_skip = h_temp
        h = torch.cat([h_temp, emb], dim=-1)
        for i, (spatial, norm) in enumerate(zip(self.spatial_layers, self.layer_norms)):
            h_in = h
            h = spatial(h, edge_index, edge_weight)
            h = norm(h)
            if i > 0: h = h + h_in
        h = h + self.skip_proj(h_skip)
        out = self.decoder(h)
        out = rearrange(out, 'b n (t f) -> b t n f', t=self.horizon, f=self.output_size if self.output_size is not None else 1)
        
        # Generate samples if mc_samples is provided
        if mc_samples is not None:
            sigma = out.std(dim=(1, 2), keepdim=True)
            noise_shape = (mc_samples, out.size(0), self.horizon, out.size(2), out.size(3))
            out = out.unsqueeze(0) + sigma * torch.randn(*noise_shape, device=out.device)
        
        return out

