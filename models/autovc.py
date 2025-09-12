"""
AutoVC implementation for singing voice conversion
Based on the AutoVC paper: https://arxiv.org/abs/1905.05879
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

class LinearNorm(nn.Module):
    """Linear layer with normalization"""
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))
    
    def forward(self, x):
        return self.linear_layer(x)

class ConvNorm(nn.Module):
    """1D Convolution with normalization"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)
        
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
    
    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class ContentEncoder(nn.Module):
    """Content encoder to extract linguistic content"""
    def __init__(self, dim_neck=config.BOTTLENECK_DIM, dim_emb=config.CONTENT_DIM, freq=config.N_MELS):
        super(ContentEncoder, self).__init__()
        
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(freq if i == 0 else config.HIDDEN_DIM,
                        config.HIDDEN_DIM,
                        kernel_size=5, stride=1,
                        padding=2,
                        dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(config.HIDDEN_DIM))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(config.HIDDEN_DIM, dim_neck, 2, batch_first=True, bidirectional=True)
        
    def forward(self, x):
        # x: (B, mel_bins, T)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        
        # Transpose for LSTM: (B, T, mel_bins)
        x = x.transpose(1, 2)
        
        # LSTM
        outputs, _ = self.lstm(x)
        
        # Take only forward direction for bottleneck
        out_forward = outputs[:, :, :self.dim_neck]
        
        return out_forward

class Decoder(nn.Module):
    """Decoder to reconstruct mel-spectrogram"""
    def __init__(self, dim_neck=config.BOTTLENECK_DIM, dim_emb=config.STYLE_DIM, dim_pre=config.HIDDEN_DIM):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck + dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                        dim_pre,
                        kernel_size=5, stride=1,
                        padding=2,
                        dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, dim_pre, 2, batch_first=True, bidirectional=True)
        
        self.linear_projection = LinearNorm(dim_pre * 2, config.N_MELS)
        
    def forward(self, x):
        # x: (B, T, dim_neck + dim_emb)
        x, _ = self.lstm1(x)
        
        # Transpose for conv: (B, dim_pre, T)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        
        # Transpose back for LSTM: (B, T, dim_pre)
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)
        
        return decoder_output

class SpeakerEncoder(nn.Module):
    """Speaker encoder to extract speaker embedding"""
    def __init__(self, c_in=config.N_MELS, c_h=config.HIDDEN_DIM, c_out=config.STYLE_DIM, kernel_size=5,
                 bank_size=8, bank_scale=1, c_bank=config.HIDDEN_DIM, n_conv_blocks=6,
                 n_dense_blocks=6, subsample_factor=4):
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample_factor = subsample_factor
        
        # Conv1d bank
        self.conv1d_bank = nn.ModuleList()
        for k in range(bank_scale, bank_size + 1, bank_scale):
            self.conv1d_bank += [nn.Conv1d(c_in, c_bank, kernel_size=k, stride=1,
                                           padding=k//2)]
        
        # Max pooling
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Conv1d projections
        self.conv1d_proj1 = nn.Conv1d(len(self.conv1d_bank) * c_bank, c_h, kernel_size=3, stride=1, padding=1)
        self.conv1d_proj2 = nn.Conv1d(c_h, c_h, kernel_size=3, stride=1, padding=1)
        self.norm_proj1 = nn.BatchNorm1d(c_h)
        self.norm_proj2 = nn.BatchNorm1d(c_h)
        
        # Highway network
        self.highway = nn.ModuleList()
        for _ in range(n_conv_blocks):
            self.highway += [HighwayNet(c_h)]
        
        # BiLSTM
        self.bilstm = nn.LSTM(c_h, c_h // 2, batch_first=True, bidirectional=True)
        
        # Dense layers
        self.dense = nn.ModuleList()
        for _ in range(n_dense_blocks):
            self.dense += [nn.Linear(c_h, c_h)]
        
        # Final projection
        self.final_proj = nn.Linear(c_h, c_out)
        
    def forward(self, x):
        # Conv1d bank
        outs = []
        for layer in self.conv1d_bank:
            out = torch.relu(layer(x))
            outs.append(out)
        out = torch.cat(outs, dim=1)
        
        # Max pooling
        out = self.max_pool1d(out)
        
        # Conv1d projections
        out = torch.relu(self.norm_proj1(self.conv1d_proj1(out)))
        out = self.norm_proj2(self.conv1d_proj2(out))
        
        # Residual connection
        out = out + x
        
        # Highway network
        for layer in self.highway:
            out = layer(out)
        
        # BiLSTM
        out = out.transpose(1, 2)  # (B, T, C)
        out, _ = self.bilstm(out)
        
        # Dense layers
        for layer in self.dense:
            out = torch.relu(layer(out))
        
        # Final projection
        out = self.final_proj(out)
        
        # Global average pooling
        out = torch.mean(out, dim=1)
        
        return out

class HighwayNet(nn.Module):
    """Highway Network"""
    def __init__(self, size):
        super(HighwayNet, self).__init__()
        self.H = nn.Linear(size, size)
        self.T = nn.Linear(size, size)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, C)
        h = torch.relu(self.H(x))
        t = torch.sigmoid(self.T(x))
        out = h * t + x * (1 - t)
        return out.transpose(1, 2)  # (B, C, T)

class AutoVC(nn.Module):
    """AutoVC model for voice conversion"""
    def __init__(self):
        super(AutoVC, self).__init__()
        
        self.content_encoder = ContentEncoder()
        self.speaker_encoder = SpeakerEncoder()
        self.decoder = Decoder()
        
    def forward(self, x_src, x_trg=None):
        """
        Forward pass
        
        Args:
            x_src: Source mel-spectrogram (B, mel_bins, T)
            x_trg: Target mel-spectrogram for speaker embedding (optional)
            
        Returns:
            Dictionary with outputs
        """
        # Extract content from source
        content = self.content_encoder(x_src)  # (B, T, dim_neck)
        
        # Extract speaker embedding
        if x_trg is not None:
            speaker_emb = self.speaker_encoder(x_trg)  # (B, dim_emb)
        else:
            # Use source speaker embedding for reconstruction
            speaker_emb = self.speaker_encoder(x_src)  # (B, dim_emb)
        
        # Expand speaker embedding to match content length
        T = content.size(1)
        speaker_emb_expanded = speaker_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, dim_emb)
        
        # Concatenate content and speaker embedding
        decoder_input = torch.cat([content, speaker_emb_expanded], dim=-1)  # (B, T, dim_neck + dim_emb)
        
        # Decode
        output = self.decoder(decoder_input)  # (B, T, mel_bins)
        
        # Transpose output to match input format
        output = output.transpose(1, 2)  # (B, mel_bins, T)
        
        return {
            'converted': output,
            'content': content,
            'speaker_emb': speaker_emb
        }

class AutoVCLoss(nn.Module):
    """Loss function for AutoVC"""
    def __init__(self, lambda_rec=1.0, lambda_content=0.1):
        super(AutoVCLoss, self).__init__()
        self.lambda_rec = lambda_rec
        self.lambda_content = lambda_content
        self.mse_loss = nn.MSELoss()
        
    def forward(self, outputs, targets):
        """
        Calculate loss
        
        Args:
            outputs: Model outputs dictionary
            targets: Target mel-spectrograms
            
        Returns:
            Total loss and loss components
        """
        # Reconstruction loss
        rec_loss = self.mse_loss(outputs['converted'], targets)
        
        # Content preservation loss (optional)
        content_loss = torch.tensor(0.0, device=outputs['converted'].device)
        
        total_loss = self.lambda_rec * rec_loss + self.lambda_content * content_loss
        
        return {
            'total_loss': total_loss,
            'rec_loss': rec_loss,
            'content_loss': content_loss
        }
