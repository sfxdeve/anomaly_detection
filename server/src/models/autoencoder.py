import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=29):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def get_reconstruction_error(self, x):
        x_tensor = torch.FloatTensor(x)
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x_tensor)
            mse = (reconstructed - x_tensor).pow(2).mean(dim=1)
        return mse.numpy()
