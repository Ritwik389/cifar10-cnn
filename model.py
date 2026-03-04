import torch
import torch.nn as nn


class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Helper: Standard Conv Block (Expensive but Powerful)
        def conv_block(in_c, out_c, kernel=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(inplace=True) # SiLU (Swish) helps deep plain nets train better
            )

        # Helper: Depthwise Separable Block (Cheap & Wide)
        def ds_conv_block(in_c, out_c, stride=1):
            return nn.Sequential(
                # Depthwise (Spatial)
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.SiLU(inplace=True),
                # Pointwise (Channel Mix)
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(inplace=True)
            )

        # --- Layers ---
        # Block 1: High Quality Feature Extraction (Standard Convs)
        # We invest heavily here: 64 channels right away.
        self.block1 = conv_block(3, 64)
        self.block2 = conv_block(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: Mid-Level Features (Standard Convs)
        # Increasing to 128 channels.
        self.block3 = conv_block(64, 128)
        self.block4 = conv_block(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: The "Brain" (Depthwise Separable)
        # We use DS Convs to explode the width to 384 channels.
        # This gives the model massive capacity to memorize patterns.
        self.block5 = ds_conv_block(128, 384)
        self.block6 = ds_conv_block(384, 384)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(384, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(self.block2(x))

        x = self.block3(x)
        x = self.pool2(self.block4(x))

        x = self.block5(x)
        x = self.pool3(self.block6(x))

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

