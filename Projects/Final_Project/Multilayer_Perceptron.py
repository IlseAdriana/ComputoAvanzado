import torch
import torch.nn as nn
from torch.utils.data import Dataset


# Convert the data into tensors, and then into a dataset
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float32, requires_grad=False)
        self.labels = torch.tensor(y, dtype=torch.int64, requires_grad=False)

    def __getitem__(self, index):
      x = self.features[index]
      y = self.labels[index]        
      return x, y

    def __len__(self):
      return self.labels.shape[0]
    
    import torch.nn as nn

# Define the Neural Network model
class MLP(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super().__init__()
        
        self.network = nn.Sequential(
            # hidden layer
            nn.Linear(num_features, hidden_size),
            nn.Sigmoid(),
            # output layer
            nn.Linear(hidden_size, num_classes)
        )
        
    
    def forward(self, x):
        return self.network(x)