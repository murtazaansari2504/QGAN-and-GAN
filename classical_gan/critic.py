import torch.nn as nn

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            # Takes [x, y] pairs as input
            nn.Linear(2, 16),  
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            # Outputs a raw, unbounded score (NO Sigmoid layer!)
            nn.Linear(16, 1)
        )
        
    def forward(self, pair):
        return self.model(pair)