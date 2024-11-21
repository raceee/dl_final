import torch

class GNN:
    def __init__(self, num_layers, num_features, num_classes):
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(num_features, 16))
        for i in range(num_layers - 1):
            self.layers.append(GCNConv(16, 16))
        self.layers.append(GCNConv(16, num_classes))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[self.num_layers](x, edge_index)
        return