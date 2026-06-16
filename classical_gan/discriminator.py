import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Takes [x, y] pairs as input
            nn.Linear(2, 16),  
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            # Outputs a probability (0 to 1)
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, pair):
        return self.model(pair)