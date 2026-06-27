"""
AutoVC implementation for singing voice conversion
Based on the AutoVC paper: https://arxiv.org/abs/1905.05879

This implementation uses an encoder/decoder with an information bottleneck
plus a code-consistency (content) loss, which is the key mechanism that lets
AutoVC perform zero-shot voice conversion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class LinearNorm(nn.Module):
    """Linear layer with Xavier initialization."""

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    """1D convolution with Xavier initialization."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)


class ContentEncoder(nn.Module):
    """Content encoder: extracts linguistic/musical content from a mel-spectrogram."""

    def __init__(self, dim_neck=config.BOTTLENECK_DIM, freq=config.N_MELS):
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

        self.lstm = nn.LSTM(config.HIDDEN_DIM, dim_neck, 2,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: (B, mel_bins, T)
        for conv in self.convolutions:
            x = F.relu(conv(x))

        # Transpose for LSTM: (B, T, hidden)
        x = x.transpose(1, 2)

        outputs, _ = self.lstm(x)

        # Take only the forward direction for the bottleneck code
        out_forward = outputs[:, :, :self.dim_neck]

        return out_forward


class Decoder(nn.Module):
    """Decoder: reconstructs a mel-spectrogram from content + speaker embedding."""

    def __init__(self, dim_neck=config.BOTTLENECK_DIM, dim_emb=config.STYLE_DIM,
                 dim_pre=config.HIDDEN_DIM):
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


class Postnet(nn.Module):
    """Postnet: 5-layer conv stack that refines the decoder output (residual)."""

    def __init__(self, n_mels=config.N_MELS, dim=config.HIDDEN_DIM,
                 n_convs=5, kernel_size=5):
        super(Postnet, self).__init__()
        padding = (kernel_size - 1) // 2

        self.convolutions = nn.ModuleList()
        # First layer: n_mels -> dim
        self.convolutions.append(nn.Sequential(
            ConvNorm(n_mels, dim, kernel_size=kernel_size, stride=1,
                     padding=padding, dilation=1, w_init_gain='tanh'),
            nn.BatchNorm1d(dim)))
        # Middle layers: dim -> dim
        for _ in range(n_convs - 2):
            self.convolutions.append(nn.Sequential(
                ConvNorm(dim, dim, kernel_size=kernel_size, stride=1,
                         padding=padding, dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(dim)))
        # Final layer: dim -> n_mels (no activation)
        self.convolutions.append(nn.Sequential(
            ConvNorm(dim, n_mels, kernel_size=kernel_size, stride=1,
                     padding=padding, dilation=1, w_init_gain='linear'),
            nn.BatchNorm1d(n_mels)))

    def forward(self, x):
        # x: (B, n_mels, T) -> residual refinement (B, n_mels, T)
        for i, conv in enumerate(self.convolutions):
            if i < len(self.convolutions) - 1:
                x = torch.tanh(conv(x))
            else:
                x = conv(x)
        return x


class SpeakerEncoder(nn.Module):
    """Speaker encoder: produces a fixed-size speaker (timbre) embedding.

    Input:  (B, n_mels, T)
    Output: (B, c_out) L2-normalized speaker embedding
    """

    def __init__(self, c_in=config.N_MELS, c_h=config.HIDDEN_DIM,
                 c_out=config.STYLE_DIM, kernel_size=5, n_conv_blocks=3):
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        padding = (kernel_size - 1) // 2

        convolutions = []
        for i in range(n_conv_blocks):
            in_ch = c_in if i == 0 else c_h
            convolutions.append(nn.Sequential(
                ConvNorm(in_ch, c_h, kernel_size=kernel_size, stride=1,
                         padding=padding, dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(c_h)))
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(c_h, c_h // 2, 1, batch_first=True, bidirectional=True)
        self.projection = LinearNorm(c_h, c_out)

    def forward(self, x):
        # x: (B, c_in, T)
        for conv in self.convolutions:
            x = F.relu(conv(x))

        # (B, T, c_h)
        x = x.transpose(1, 2)
        outputs, _ = self.lstm(x)

        # Temporal average pooling -> (B, c_h)
        pooled = torch.mean(outputs, dim=1)

        # Project and L2-normalize
        emb = self.projection(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


class AutoVC(nn.Module):
    """AutoVC model for voice conversion."""

    def __init__(self):
        super(AutoVC, self).__init__()

        self.content_encoder = ContentEncoder()
        self.speaker_encoder = SpeakerEncoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def forward(self, x_src, x_trg=None):
        """Forward pass.

        Args:
            x_src: Source mel-spectrogram (B, mel_bins, T)
            x_trg: Target mel-spectrogram for the speaker embedding (optional).
                   If None, the source's own speaker embedding is used
                   (i.e. reconstruction / autoencoding).

        Returns:
            Dictionary with:
                converted:          decoder output, pre-postnet  (B, mel_bins, T)
                converted_postnet:  postnet-refined output       (B, mel_bins, T)
                content:            source content code          (B, T, dim_neck)
                content_recon:      content code re-encoded from
                                    the converted output         (B, T, dim_neck)
                speaker_emb:        speaker embedding             (B, style_dim)
        """
        # Extract content from source
        content = self.content_encoder(x_src)  # (B, T, dim_neck)

        # Extract speaker (timbre) embedding from target (or source)
        speaker_src = x_trg if x_trg is not None else x_src
        speaker_emb = self.speaker_encoder(speaker_src)  # (B, style_dim)

        # Expand speaker embedding across time and concatenate with content
        T = content.size(1)
        speaker_emb_expanded = speaker_emb.unsqueeze(1).expand(-1, T, -1)
        decoder_input = torch.cat([content, speaker_emb_expanded], dim=-1)

        # Decode to mel-spectrogram
        mel_outputs = self.decoder(decoder_input)          # (B, T, mel_bins)
        mel_outputs = mel_outputs.transpose(1, 2)          # (B, mel_bins, T)

        # Postnet residual refinement
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)

        # Code-consistency: re-encode content from the refined output
        content_recon = self.content_encoder(mel_outputs_postnet)

        return {
            'converted': mel_outputs,
            'converted_postnet': mel_outputs_postnet,
            'content': content,
            'content_recon': content_recon,
            'speaker_emb': speaker_emb,
        }


class AutoVCLoss(nn.Module):
    """AutoVC training objective.

    total = lambda_rec * (recon_pre + recon_post) + lambda_content * code_loss

    where:
        recon_pre   = MSE(decoder output, target mel)
        recon_post  = MSE(postnet output, target mel)
        code_loss   = L1(content code of output, content code of input)
    """

    def __init__(self, lambda_rec=1.0, lambda_content=1.0):
        super(AutoVCLoss, self).__init__()
        self.lambda_rec = lambda_rec
        self.lambda_content = lambda_content
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, outputs, targets):
        """Calculate loss.

        Args:
            outputs: Model output dictionary from AutoVC.forward
            targets: Target mel-spectrograms (B, mel_bins, T)

        Returns:
            Dictionary with total_loss and individual components.
        """
        # Reconstruction losses (before and after the postnet)
        rec_loss = self.mse_loss(outputs['converted'], targets)
        rec_psnt_loss = self.mse_loss(outputs['converted_postnet'], targets)

        # Content (code-consistency) loss
        content_loss = self.l1_loss(outputs['content_recon'], outputs['content'])

        total_loss = (self.lambda_rec * (rec_loss + rec_psnt_loss)
                      + self.lambda_content * content_loss)

        return {
            'total_loss': total_loss,
            'rec_loss': rec_loss,
            'rec_psnt_loss': rec_psnt_loss,
            'content_loss': content_loss,
        }
