
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
import torch.nn.functional as F

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

            # Layers
            self.pi = nn.Linear(in_features, num_gaussians)
            self.sigma = nn.Linear(in_features, out_features * num_gaussians)
            self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        # minibatch shape: [B, T, D]
        pi = self.pi(minibatch)  # Apply linear layer along the last dimension
        sigma = torch.nn.functional.softplus(self.sigma(minibatch))
        sigma = sigma + 1e-6  # Add a small epsilon
        mu = self.mu(minibatch)

        # Reshape sigma and mu to separate out the Gaussian components
        B, T, _ = sigma.shape
        sigma = sigma.view(B, T, self.num_gaussians, self.out_features)
        mu = mu.view(B, T, self.num_gaussians, self.out_features)

        # Apply softmax to pi across the Gaussian dimension
        pi = F.softmax(pi, dim=-1)

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
    """
    Computes the Mixture Density Network loss, adapted to be more similar to Code A.

    :param pi: Mixture component weights [B, T, G]
    :param sigma: Standard deviations of mixture components [B, T, G, O]
    :param mu: Means of mixture components [B, T, G, O]
    :param target: Ground truth/target tensor [B, T, O]
    :return: loss: Scalar loss tensor
    """
    # Expand target to have G dimension for compatibility with pi, sigma, and mu
    target = target.unsqueeze(2).expand_as(mu)
    
    # Calculate the probability density function of target
    # Using Normal distribution with provided mu and sigma
    normal_dist = torch.distributions.Normal(mu, sigma)
    prob_density = normal_dist.log_prob(target)
    
    # Log-sum-exp trick for numerical stability and to align with Code A logic
    # This step is crucial to avoid underflow in computing log probabilities
    max_prob_density = prob_density.max(dim=2, keepdim=True)[0]
    prob_density = prob_density - max_prob_density
    prob_density = torch.exp(prob_density) * pi.unsqueeze(-1)
    prob_density = torch.log(prob_density.sum(2) + 1e-6) + max_prob_density.squeeze(2)

    # Negative log-likelihood
    nll = -prob_density.mean()

    return nll


def sample(pi, sigma, mu):
    # pi: Mixing coefficients with shape [B, T, G]
    # sigma: Standard deviations of the Gaussians [B, T, G, O]
    # mu: Means of the Gaussians [B, T, G, O]
    # :return: Sampled values from the MDN
    
    categorical = Categorical(pi)
    mixture_indices = categorical.sample().unsqueeze(-1)  # Add an extra dimension for gather # [B, T, 1]
    
    # Gather the chosen mixture components' parameters
    chosen_sigma = torch.gather(sigma, 2, mixture_indices.unsqueeze(-1).expand(-1, -1, -1, sigma.size(-1)))
    chosen_mu = torch.gather(mu, 2, mixture_indices.unsqueeze(-1).expand(-1, -1, -1, mu.size(-1)))
    
    # Remove the extra G dimension since we have selected the component
    chosen_sigma = chosen_sigma.squeeze(2)
    chosen_mu = chosen_mu.squeeze(2)
    
    # Sample from the normal distributions
    normal = torch.distributions.Normal(chosen_mu, chosen_sigma)
    samples = normal.sample()  # [B, T, O]
    
    return samples

