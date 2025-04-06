import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=26):
        super(EmbeddingClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x
