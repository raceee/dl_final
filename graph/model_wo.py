import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# Define the Graph Neural Network model
class GraphNN_Model_WO(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=2):
        super(GraphNN_Model_WO, self).__init__()

        self.linears = torch.nn.ModuleList()
        self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden_layers - 1):
            self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))

        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim

    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index
        x = x.float()

        for linear in self.linears:
            x = linear(x)
            x = F.relu(x)

        graph_embeddings = x.mean(dim=0, keepdim=True)
        if return_embeddings:
            return graph_embeddings

        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        
        return x