import torch.nn as nn

# Size of the random noise input vector
LATENT_DIM = 5

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            # Outputs fake [x, y] coordinates
            nn.Linear(16, 2)  
        )
        
    def forward(self, z):
        return self.model(z)