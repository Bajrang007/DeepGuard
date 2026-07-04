"""
DeepGuard video deepfake detection model: EfficientNet-B0 (per-frame feature
extractor) + GRU (temporal sequence modeling) + classification head.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DeepfakeVideoModel(nn.Module):
    def __init__(self, hidden_size=256, num_gru_layers=1, num_classes=2, pretrained=True):
        super().__init__()

        # --- Frame-level feature extractor ---
        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Remove the final classification layer - we only want the feature extractor
        self.feature_extractor = efficientnet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 1280  # EfficientNet-B0's output feature size

        # --- Temporal sequence modeling ---
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=False,
        )

        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        x shape: (batch, sequence_length, C, H, W)
        """
        batch_size, seq_len, c, h, w = x.shape

        # Flatten batch and sequence dims so EfficientNet processes all frames at once
        x = x.view(batch_size * seq_len, c, h, w)

        features = self.feature_extractor(x)          # (batch*seq_len, 1280, H', W')
        features = self.pool(features)                # (batch*seq_len, 1280, 1, 1)
        features = features.view(batch_size, seq_len, self.feature_dim)  # (batch, seq_len, 1280)

        gru_out, _ = self.gru(features)                # (batch, seq_len, hidden_size)
        last_output = gru_out[:, -1, :]                 # take the final timestep's output

        logits = self.classifier(last_output)           # (batch, num_classes)
        return logits


if __name__ == "__main__":
    # Quick sanity check: does a forward pass work with the correct shapes?
    model = DeepfakeVideoModel()
    dummy_input = torch.randn(2, 16, 3, 224, 224)  # batch of 2, 16-frame sequences
    output = model(dummy_input)
    print("Output shape:", output.shape)  # expect (2, 2) - batch of 2, 2 classes