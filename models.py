import torch
import torch.nn as nn


# --- CNN Block ---
class CNN1dBlock(nn.Module):
    def __init__(self, input_size=144, out_channels=128, kernel_size=3, dropout=0.1, use_pooling=True):
        super().__init__()
        layers = [
            nn.Conv1d(in_channels=input_size, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        ]
        if use_pooling:
            layers.append(nn.MaxPool1d(kernel_size=2))
        layers.append(nn.Dropout(dropout))
        self.cnn_block = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, F, T)
        return self.cnn_block(x)

class TemporalConvTransformer(nn.Module):
    def __init__(self, input_dim=144, patch_size=4, num_patches=4, embed_dim=256, num_heads=4,
                 num_layers=2, num_classes=100, dropout=0.1, use_seq_final_block=False):
        super().__init__()
        assert patch_size * num_patches == 16, "Ensure patching covers entire sequence (e.g., 4x4=16 frames)"

        self.patch_size = patch_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # 1D conv acts as local temporal feature extractor
        # self.temporal_conv = nn.Sequential(
        #     nn.Conv1d(in_channels=input_dim, out_channels=embed_dim, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # )

        self.temporal_conv = CNN1dBlock(input_size=input_dim, out_channels=embed_dim, dropout=dropout, use_pooling=False)

        # Linear projection of patches
        self.patch_embedding = nn.Linear(embed_dim * patch_size, embed_dim)

        # Positional encoding for patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classifier
        if use_seq_final_block:
            self.cls_head = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        else:
            self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # x: (B, T, F)
        B, T, F = x.size()
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.temporal_conv(x)  # (B, embed_dim, T)
        x = x.permute(0, 2, 1)  # (B, T, embed_dim)

        # Temporal patching: split into N patches of size `patch_size`
        x = x.reshape(B, -1, self.patch_size * self.embed_dim)  # (B, num_patches, embed_dim * patch_size)
        x = self.patch_embedding(x)  # (B, num_patches, embed_dim)

        x = x + self.pos_embed  # Add positional encoding
        x = self.transformer(x)  # (B, num_patches, embed_dim)

        x = x.mean(dim=1)  # Global average pooling over patches
        return self.cls_head(x)  # (B, num_classes)
    

class TemporalConvTransformer_B(nn.Module):
    def __init__(
        self,
        input_dim=144,
        patch_size=4,
        num_patches=4,
        embed_dim=256,
        num_heads=4,
        num_layers=2,
        num_classes=100,
        dropout=0.1,
        use_seq_final_block=False,
        use_cls_token=False
    ):
        super().__init__()
        seq_len = patch_size * num_patches
        assert seq_len == 16, (
            f"patch_size * num_patches must equal sequence length, got {patch_size}*{num_patches}={seq_len}"
        )

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.num_patches = num_patches

        # 1D conv to extract local temporal features from keypoints
        self.temporal_conv = CNN1dBlock(
            input_size=input_dim,
            out_channels=embed_dim,
            kernel_size=3,
            dropout=dropout,
            use_pooling=False
        )

        # Linear projection of concatenated patch features
        self.patch_embedding = nn.Linear(embed_dim * patch_size, embed_dim)

        # CLS token param
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            pos_count = num_patches + 1
        else:
            pos_count = num_patches
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, pos_count, embed_dim))

        # Transformer layers with post-layernorm
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            encoder = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            norm = nn.LayerNorm(embed_dim)
            self.transformer_layers.append(nn.ModuleDict({'encoder': encoder, 'norm': norm}))

        # Classification head
        if use_seq_final_block:
            self.cls_head = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        else:
            self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: (B, T, F) where T=16, F=input_dim
        """
        B, T, F = x.size()
        # Temporal Conv (B, F, T) -> (B, embed_dim, T)
        x = x.permute(0, 2, 1)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)  # (B, T, embed_dim)

        # Split into patches and embed
        x = x.reshape(B, self.num_patches, self.patch_size * self.embed_dim)
        x = self.patch_embedding(x)  # (B, num_patches, embed_dim)

        # Prepend CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer stack with post-LayerNorm
        for layer in self.transformer_layers:
            x = layer['encoder'](x)
            x = layer['norm'](x)

        # Classification: use CLS token or mean pooling
        if self.use_cls_token:
            x = x[:, 0]  # CLS token
        else:
            x = x.mean(dim=1)
        return self.cls_head(x)



# --- CNN + RNN (GRU/LSTM) Hybrid ---
class CNNRNN(nn.Module):
    def __init__(self, input_size=144, num_classes=100,  rnn_hidden_size=128, num_cnn_blocks=2, use_lstm=False, **kwargs):
        super().__init__()

        cnn_kwargs = kwargs.get('cnn_kwargs', {})
        rnn_kwargs = kwargs.get('rnn_kwargs', {})

        # Initial CNN block input channel
        channels = [input_size] + [cnn_kwargs.get('out_channels', 128)] * num_cnn_blocks

        self.cnn_blocks = nn.Sequential(*[
            CNN1dBlock(
                input_size=channels[i],
                out_channels=channels[i+1],
                kernel_size=cnn_kwargs.get('kernel_size', 3),
                dropout=cnn_kwargs.get('dropout', 0.1),
                use_pooling=cnn_kwargs.get('use_pooling', True)
            ) for i in range(num_cnn_blocks)
        ])

        rnn_input_size = channels[-1]  # Final output channels from last CNN
        rnn = nn.LSTM if use_lstm else nn.GRU
        self.rnn = rnn(
            rnn_input_size,
            rnn_hidden_size,
            batch_first=True,
            **rnn_kwargs
        )

        direction_factor = 2 if rnn_kwargs.get('bidirectional', False) else 1
        self.fc = nn.Linear(rnn_hidden_size * direction_factor, num_classes)

    def forward(self, x):  # x: (B, T, F)
        x = x.permute(0, 2, 1)         # (B, F, T)
        x = self.cnn_blocks(x)         # (B, C, T')
        x = x.permute(0, 2, 1)         # (B, T', C)
        out, _ = self.rnn(x)           # (B, T', H*2)
        out = out[:, -1, :]            # Last timestep
        return self.fc(out)


# --- GRU Only ---
class GRUModel(nn.Module):
    def __init__(self, input_size=144, num_classes=100, rnn_hidden_size=128, **kwargs):
        super().__init__()
        rnn_kwargs = kwargs.get('rnn_kwargs', {})
        self.gru = nn.GRU(input_size, rnn_hidden_size, batch_first=True, **rnn_kwargs)
        direction_factor = 2 if rnn_kwargs.get('bidirectional', False) else 1
        self.fc = nn.Linear(rnn_hidden_size * direction_factor, num_classes)

    def forward(self, x):  # (B, T, F)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


# --- LSTM Only ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=144, num_classes=100, rnn_hidden_size=128, **kwargs):
        super().__init__()
        rnn_kwargs = kwargs.get('rnn_kwargs', {})
        self.lstm = nn.LSTM(input_size, rnn_hidden_size, batch_first=True, **rnn_kwargs)
        direction_factor = 2 if rnn_kwargs.get('bidirectional', False) else 1
        self.fc = nn.Linear(rnn_hidden_size * direction_factor, num_classes)

    def forward(self, x):  # (B, T, F)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class BiGRUWithSeq(nn.Module):
    def __init__(self, input_size=144, hidden_size=256, num_layers=1, num_classes=100, bidirectional=True, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        direction_factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * direction_factor, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):  # x: (B, T, F) = (batch, frames, keypoints*3)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])
    

class CNNBiGRU_diogo(nn.Module):
    def __init__(self, input_size=144, hidden_size=256, num_layers=1, num_classes=100):
        super().__init__()

        # CNN: aprende padrões espaciais entre keypoints por frame
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),  # (B, 64, T)
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1)
        )

        # GRU: aprende dependências temporais
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Classificação final
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):  # x: (B, T, F)
        x = x.permute(0, 2, 1)          # (B, F, T) → necessário para Conv1d
        x = self.cnn(x)                 # (B, 64, T)
        x = x.permute(0, 2, 1)          # (B, T, 64) → para GRU
        out, _ = self.gru(x)            # (B, T, H*2)
        out = out[:, -1, :]             # último frame
        return self.fc(out)             # (B, num_classes)




# ============================================= TRASH =========================================
'''
import torch
import torch.nn as nn
from fastai.vision.all import *

class BiGRUWrapper(nn.Module):
    def __init__(self, input_size=144, hidden_size=256, num_layers=1, num_classes=100, bidirectional=True, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        direction_factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * direction_factor, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):  # x: (B, T, F) = (batch, frames, keypoints*3)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])

class SimpleBiGRU(nn.Module):
    def __init__(self, input_size=144, hidden_size=256, num_layers=1, num_classes=100):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 por ser bidirecional

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.gru(x)              # out: (B, T, H*2)
        out = out[:, -1, :]               # só último frame
        out = self.flatten(out)           # (B, H*2)
        return self.fc(out)               # (B, num_classes)

class CNNBiGRU(nn.Module):
    def __init__(self, input_size=144, hidden_size=256, num_layers=1, num_classes=100):
        super().__init__()

        # CNN: aprende padrões espaciais entre keypoints por frame
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),  # (B, 64, T)
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1)
        )

        # GRU: aprende dependências temporais
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Classificação final
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):  # x: (B, T, F)
        x = x.permute(0, 2, 1)          # (B, F, T) → necessário para Conv1d
        x = self.cnn(x)                 # (B, 64, T)
        x = x.permute(0, 2, 1)          # (B, T, 64) → para GRU
        out, _ = self.gru(x)            # (B, T, H*2)
        out = out[:, -1, :]             # último frame
        return self.fc(out)             # (B, num_classes)
    
'''