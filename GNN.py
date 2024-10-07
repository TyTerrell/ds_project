import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import joblib

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # first Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        return x

    def save_model(self, path='gnn_model.pth'):
        """
        saving the model using joblib's built in function.
        """
        joblib.dump(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path='gnn_model.pth'):
        """
        same as the saving, loading the model with the function.
        """
        self.load_state_dict(joblib.load(path))
        self.eval()  # this lets the model be in evaluation mode, for deployment.
        print(f"Model loaded from {path}")