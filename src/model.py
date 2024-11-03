import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch_geometric.nn import GATConv

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes, num_landmarks=136, gat_out_dim=64):
        super(EmotionClassifier, self).__init__()
        
        # ResNet Backbone
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Identity()
        
        # Graph Network (Multi-Head GAT)
        self.gat1 = GATConv(2, gat_out_dim, heads=4, concat=True)  # Input features are (68, 2)
        self.gat2 = GATConv(gat_out_dim * 4, gat_out_dim, heads=4, concat=True)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # Final classifier
        self.fc_out = nn.Linear(gat_out_dim * 4 + 512, num_classes)  # 512 for image features from ResNet

    def forward(self, image, landmarks, edge_index):
        # Image features
        image_features = self.resnet(image)  # Shape: (batch_size, 512)
        
        # Process each item in the batch separately for GAT
        gat_outputs = []
        for i in range(landmarks.size(0)):
            # Copy edge_index for individual graph processing
            individual_edge_index = edge_index[i]  # Get edge index for the specific batch item
            
            # Pass through GAT layers for each sample in batch
            x = F.relu(self.gat1(landmarks[i], individual_edge_index))  # Input: (68, 2)
            x = self.dropout(F.relu(self.gat2(x, individual_edge_index)))
            gat_outputs.append(x.mean(dim=0, keepdim=True))  # Global mean pooling for each graph
            
        # Stack the processed landmark features
        gat_outputs = torch.cat(gat_outputs, dim=0)  # Shape: (batch_size, gat_out_dim)

        # Combine GAT output with image features
        x = torch.cat([gat_outputs, image_features], dim=1)

        # Final classification
        x = self.fc_out(x)
        return x
