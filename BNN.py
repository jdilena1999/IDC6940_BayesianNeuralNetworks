import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from data_processing import X_transformed,y

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Learnable mean and (log-scale) std for each weight -- this IS the "distribution" over weights
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))  # rho -> std via softplus
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

    def forward(self, x):
        # Reparameterization trick: sample weights as mu + std * noise (differentiable)
        weight_std = F.softplus(self.weight_rho)
        bias_std = F.softplus(self.bias_rho)
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        return F.linear(x, weight, bias)

    def kl_loss(self):
        # KL divergence between learned weight distribution and a standard normal prior N(0,1)
        weight_std = F.softplus(self.weight_rho)
        bias_std = F.softplus(self.bias_rho)
        kl = 0.5 * torch.sum(weight_std**2 + self.weight_mu**2 - 1 - 2*torch.log(weight_std))
        kl += 0.5 * torch.sum(bias_std**2 + self.bias_mu**2 - 1 - 2*torch.log(bias_std))
        return kl

class BayesianNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = BayesianLinear(in_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def kl_loss(self):
        return self.fc1.kl_loss() + self.fc2.kl_loss()

# Training loop
model = BayesianNN(in_dim=784, hidden_dim=64, out_dim=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
kl_weight = 1e-4  # controls how strongly the prior regularizes toward N(0,1)

num_epochs = 1_000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x_batch)
    nll = F.cross_entropy(output, y_batch)          # data-fit term
    loss = nll + kl_weight * model.kl_loss()         # + prior regularization term
    loss.backward()
    optimizer.step()

# Inference: sample multiple forward passes to get predictive uncertainty
preds = torch.stack([model(x_test) for _ in range(100)])
mean_pred = preds.mean(dim=0)
uncertainty = preds.std(dim=0)   # this is the payoff -- per-prediction