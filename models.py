import torch
import sys
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
import numpy as np
from Conformer import ConformerBlock


class ResLSTM(nn.Module):
    def __init__(self, dimension: int,
                num_layers: int = 2,
                bidirectional: bool = False,
                skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension if not bidirectional else dimension // 2,
                            num_layers, batch_first=True,
                            bidirectional=bidirectional)

    def forward(self, x):
        """
        Args:
            x: [B, F, T]

        Returns:
            y: [B, F, T]
        """
        # x = rearrange(x, "b f t -> b t f")
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        # y = rearrange(y, "b t f -> b f t")
        return y

class AudioTokenEmbedding(nn.Module):
    def __init__(self, num_quantizers=8, vocab_size=1024, embedding_dim=256):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for _ in range(num_quantizers)
        ])
    
    def forward(self, tokens):
        """
        tokens: Tensor of shape (B, Q, T), values in [0, vocab_size - 1]
        Returns: Tensor of shape (Q, B, T, embedding_dim)
        """
        B, Q, T = tokens.shape
        embedding_outputs = []
        for q in range(Q):
            emb = self.embeddings[q](tokens[:, q, :])  # (B, T, embedding_dim)
            embedding_outputs.append(emb) # (B, T, embedding_dim)
        return embedding_outputs

class ChannelLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):          # x: [B, C, T]
        return self.ln(x.transpose(1, 2)).transpose(1, 2)

class Downsampling8x(nn.Module):
    def __init__(self, in_channels, channels_1=855, channels_2=684, out_channels=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,  
                out_channels=channels_1,
                kernel_size=3,
                stride=2,
                padding=1  
            ),
            ChannelLayerNorm(channels_1),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(
                in_channels=channels_1,
                out_channels=channels_2,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            ChannelLayerNorm(channels_2),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(
                in_channels=channels_2,
                out_channels=out_channels,  
                kernel_size=3,
                stride=2,
                padding=1
            ),
            ChannelLayerNorm(out_channels),
            nn.LeakyReLU(0.2)
            
        )
        
    def forward(self, x):
        return self.layers(x)

class NoisyBranch(torch.nn.Module):
    def __init__(self, input_dim, project_dim, output_dim, num_blocks=8):
        super().__init__()
        self.num_blocks = num_blocks
        self.downsample = Downsampling8x(in_channels=input_dim)
        self.lstm = ResLSTM(project_dim, num_layers=2, bidirectional=True)
        self.conformer = nn.ModuleList([])
        for i in range(num_blocks):
            self.conformer.append(ConformerBlock(project_dim, dilation=2 ** (i % 4)))
        self.norm = nn.LayerNorm(project_dim, eps=1e-6)
        self.output_project = nn.Linear(project_dim, output_dim)
    
    def forward(self, x):
        x = self.downsample(x)
        x = x.permute(0,2,1)
        x = self.lstm(x)
        for i in range(self.num_blocks): 
            x = self.conformer[i](x) + x
            x = self.norm(x)
        x = self.output_project(x)
        return x

class TokenGenerator(nn.Module):
    def __init__(self, code_size=1024, input_dim=32, mid_dim=512, num_blocks=4):
        super().__init__()
        self.num_blocks = num_blocks
        self.project = nn.Linear(2*mid_dim, mid_dim)
        self.lstm = ResLSTM(mid_dim, num_layers=2, bidirectional=True)
        self.conformer = nn.ModuleList([])
        for i in range(num_blocks):
            self.conformer.append(ConformerBlock(mid_dim, dilation=2 ** (i % 4)))
        self.norm = nn.LayerNorm(mid_dim, eps=1e-6)
        self.proj = nn.Linear(mid_dim, code_size)

    def forward(self, x, dac):
        x = torch.cat((x, dac),-1)
        x = self.project(x)
        x = self.lstm(x)
        for i in range(self.num_blocks):      
            x = self.conformer[i](x) + x
            x = self.norm(x)
        x = self.proj(x)
        
        return x

class ParaGSE(nn.Module):
    def __init__(self, code_size=256, input_dim=513*2, project_dim=512, mid_dim=512, num_quantize=4, code_dim=32, num_blocks=1):
        super().__init__()
        self.num_quantize = num_quantize
        self.code_dim = code_dim
        self.embed = AudioTokenEmbedding(num_quantizers=num_quantize, vocab_size=code_size, embedding_dim=mid_dim)
        self.branch = NoisyBranch(input_dim=input_dim, project_dim=project_dim, output_dim=mid_dim, num_blocks=num_blocks)
        self.backbone = nn.ModuleList([])
        for i in range(num_quantize):
            self.backbone.append(TokenGenerator(code_size=code_size, input_dim=code_dim//num_quantize, mid_dim=mid_dim, num_blocks=num_blocks))
    
    def forward(self, x, dac):
        quan_list = []
        dac = self.branch(dac)
        x = self.embed(x)
        for i in range(self.num_quantize):
            y = self.backbone[i](x[i], dac)
            quan_list.append(y)
        return quan_list

def main():
    x = []
    for i in range(9):
        y = torch.randint(low=0, high=1023, size=(4,161,1))
        x.append(y)
    cond1 = torch.randn(4,161,1024)
    dac = torch.randn(4,161,9*8)
    model = ParaGSE()
    x = model(x, dac, cond1)
    print(x[0].size())

if __name__ == "__main__":
    main()