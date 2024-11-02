import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch_geometric.nn import GATConv, GCNConv


# TODO - This is not working
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes, num_landmarks=136, gat_out_dim=64, hidden_dim=128):
        super(EmotionClassifier, self).__init__()
        
        # ResNet Backbone
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Identity()
        
        # Fully Connected Network for ResNet features
        self.fcn = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Landmark processing - Adjusting input size to 136
        self.landmark_fc = nn.Linear(num_landmarks, hidden_dim // 2)
        
        # Graph Network (GCN + Multi-Head GAT)
        self.gcn = GCNConv(hidden_dim + (hidden_dim // 2), hidden_dim)  # Update input size here
        self.gat1 = GATConv(hidden_dim, gat_out_dim, heads=4, concat=True)
        self.gat2 = GATConv(gat_out_dim * 4, gat_out_dim, heads=4, concat=True)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        # Final classifier
        self.fc_out = nn.Linear(gat_out_dim * 4, num_classes)

    def forward(self, image, landmarks, edge_index):
        # Image features
        image_features = self.resnet(image)
        image_features = self.fcn(image_features)
        
        # Landmark features
        landmarks = F.relu(self.landmark_fc(landmarks))  # This should be of shape (batch_size, hidden_dim // 2)
        
        # Concatenate image and landmark features
        combined_features = torch.cat([image_features, landmarks], dim=1)  # shape will be (batch_size, hidden_dim + hidden_dim // 2)
        
        # Graph Network
        x = self.gcn(combined_features, edge_index)
        x = F.relu(self.gat1(x, edge_index))
        x = self.dropout(F.relu(self.gat2(x, edge_index)))
        
        # Final classification
        x = self.fc_out(x)
        return x
