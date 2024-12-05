import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# Define the Graph Neural Network model
class GraphNN_Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GraphNN_Model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids):
        x, edge_index, batch = input_ids.x, input_ids.edge_index, input_ids.batch
        x = x.float()

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.fc(x)
        # x = F.log_softmax(x, dim=1)
        
        return x