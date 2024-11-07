import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class HourglassBlock(nn.Module):
    def __init__(self, in_channels, depth=4):
        super(HourglassBlock, self).__init__()
        self.depth = depth
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        # Downsample layers
        for _ in range(depth):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            ))
        
        # Upsample layers
        for _ in range(depth):
            self.upsample_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            ))

    def forward(self, x):
        skip_connections = []
        
        # Downsampling path
        for down in self.downsample_layers:
            x = down(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.downsample_layers[-1](x)
        
        # Upsampling path with skip connections
        for up in self.upsample_layers:
            skip_connection = skip_connections.pop()
            # Ensure skip_connection matches the size of x before addition
            if skip_connection.size() != x.size():
                skip_connection = F.interpolate(skip_connection, size=x.size()[2:], mode='bilinear', align_corners=False)
            x = up(x + skip_connection)
        
        return x


class EmotionClassifier(nn.Module):
    def __init__(self, num_classes, num_landmarks=136, gat_out_dim=64):
        super(EmotionClassifier, self).__init__()
        
        # Hourglass Backbone
        self.hourglass = HourglassBlock(in_channels=64, depth=4)
        
        # Initial convolution for grayscale images
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Graph Network (Multi-Head GAT)
        self.gat1 = GATConv(3, gat_out_dim, heads=4, concat=True)
        self.gat2 = GATConv(gat_out_dim * 4, gat_out_dim, heads=4, concat=True)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # Final classifier
        self.fc_out = nn.Linear(gat_out_dim * 4 + 64, num_classes)

    def forward(self, image, landmarks, edge_index):
        # Image features through Hourglass network
        x = self.initial_conv(image)
        image_features = self.hourglass(x)
        image_features = F.adaptive_avg_pool2d(image_features, (1, 1)).view(image_features.size(0), -1)
        # Process each item in the batch separately for GAT
        gat_outputs = []
        for i in range(landmarks.size(0)):
            individual_edge_index = edge_index[i]
            x = F.relu(self.gat1(landmarks[i], individual_edge_index))
            x = self.dropout(F.relu(self.gat2(x, individual_edge_index)))
            gat_outputs.append(x.mean(dim=0, keepdim=True))
            
        # Stack the processed landmark features
        gat_outputs = torch.cat(gat_outputs, dim=0)
        
        # Combine GAT output with image features
        x = torch.cat([gat_outputs, image_features], dim=1)

        # Final classification
        x = self.fc_out(x)
        return x
