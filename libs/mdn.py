
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
            nn.Linear(in_features, num_gaussians),  # Softmax will be applied later
            nn.Softmax(dim=-1)  # Assuming the softmax is applied across the Gaussian dimension
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians, device=device)
        self.mu = nn.Linear(in_features, out_features * num_gaussians,device=device)

    def forward(self, minibatch):
        B, T, D = minibatch.shape
        minibatch_flat = minibatch.view(-1, D)  # Flatten [B, T, D] to [B*T, D]

        pi_flat = self.pi(minibatch_flat)
        sigma_flat = self.sigma(minibatch_flat)
        sigma_flat = torch.nn.functional.softplus(self.sigma(minibatch_flat))
        sigma_flat = sigma_flat + 1e-6  # Add a small epsilon to ensure positivity
        mu_flat = self.mu(minibatch_flat)

        pi = pi_flat.view(B, T, -1)
        sigma = sigma_flat.view(B, T, self.num_gaussians, self.out_features)
        mu = mu_flat.view(B, T, self.num_gaussians, self.out_features)

        # Apply softmax over the correct dimension for pi
        pi = nn.functional.softmax(pi, dim=-1)  # Assuming the output of self.pi is [B, T, G]

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
    Computes the Mixture Density Network loss.
    
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
    prob_density = torch.exp(normal_dist.log_prob(target))
    
    pi = pi.unsqueeze(-1)
    # Multiply by the mixture probabilities pi
    prob_density = prob_density * pi
    
    # Sum over the mixture dimension G and take log
    prob_density = prob_density.sum(2)
    nll = -torch.log(prob_density + 1e-6)  # Numerical stability
    
    # Take mean over batch and time dimensions B and T
    loss = nll.mean()
    
    return loss

def sample(pi, sigma, mu):
    # pi: Mixing coefficients with shape [B, T, G]
    # sigma: Standard deviations of the Gaussians [B, T, G, O]
    # mu: Means of the Gaussians [B, T, G, O]
    
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

