
# taken from  https://github.com/sagelywizard/pytorch-mdn
# put here because install was unsuccessful

# adapted to work with 3D data

"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
import math
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxTxD): B is the batch size, T is the sequence length, and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxTxG, BxTxGxO, BxTxGxO): B is the batch size, T is the sequence length, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians, device=device),
            nn.Softmax(dim=1)  # Change to dim=1 as we will process the flattened input
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians, device=device)
        self.mu = nn.Linear(in_features, out_features * num_gaussians, device=device)

    def forward(self, minibatch):
        B, T, D = minibatch.shape
        minibatch_flat = minibatch.view(-1, D)  # Flatten [B, T, D] to [B*T, D]

        pi_flat = nn.functional.softmax(self.pi(minibatch_flat), dim=1)
        # change to softplus because input is normalised between -1 and 1
        sigma_flat = torch.nn.functional.softplus(self.sigma(minibatch_flat))
        mu_flat = self.mu(minibatch_flat)

        pi = pi_flat.view(B, T, -1)
        sigma = sigma_flat.view(B, T, self.num_gaussians, self.out_features)
        mu = mu_flat.view(B, T, self.num_gaussians, self.out_features)

        return pi, sigma, mu


def gaussian_probability(pi, sigma, mu, target):
    sigma = sigma + 1e-6  # Add epsilon for numerical stability
    target = target.unsqueeze(2).expand_as(sigma)
    log_ret = -0.5 * ((target - mu) / sigma)**2 - torch.log(sigma) - 0.5 * torch.log(torch.tensor(2 * math.pi, device=sigma.device, dtype=sigma.dtype))

    # Apply log-sum-exp trick for numerical stability
    m, _ = torch.max(log_ret, dim=3, keepdim=True)
    sum_exp = torch.sum(torch.exp(log_ret - m), dim=3, keepdim=True)
    log_sum = m + torch.log(sum_exp)
    
    # Combine with log of pi and exponentiate
    log_prob = torch.log(pi) + log_sum.squeeze(3)
    return torch.exp(log_prob)



def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    # this is where it all goes wrong - the probabilities are all 0
    prob = gaussian_probability(pi, sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=2))
    return torch.mean(nll)

def sample(pi, sigma, mu):
    B, T, G, O = sigma.shape
    pis = Categorical(pi).sample().view(B, T, 1, 1)
    gaussian_noise = torch.randn((B, T, O), requires_grad=False, device=device)
    variance_samples = sigma.gather(2, pis).squeeze(2)
    mean_samples = mu.gather(2, pis).squeeze(2)
    return gaussian_noise * variance_samples + mean_samples
