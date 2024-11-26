import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# Define the Graph Neural Network model
class GraphNN_Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNN_Model, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc(x)
        
        return x