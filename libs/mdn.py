
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
from torch.distributions import Normal

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


def random_sample(pi, sigma, mu):
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


def sample(pi, sigma, mu , variance_div= 100):
    # CHANGE: Instead of random sampling, use the mean of the most probable component
    alpha_idx = torch.argmax(pi, dim=2)  # Find the index of the most probable Gaussian component
    selected_mu = mu.gather(2, alpha_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1,-1, mu.size(-1)))
    selected_sigma = sigma.gather(2, alpha_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, sigma.size(-1)))
    selected_sigma = selected_sigma/ variance_div
    
    # print(variance_div)
    # Divide by 100 to reduce the variance of the selected Gaussian I think, Pette et al 2019 did this not sure why
    # but it seems to help model be less jaggy? - sigma is variance, so smaller sigma is closer to mean - less jaggy but more boring
    
    normal_dist = Normal(selected_mu, selected_sigma)
    next_values = normal_dist.sample()[:, -1, :]  # Sample from the selected Gaussian
    
    # random sample - previous implementation
    # next_values = mdn.sample(pi, sigma, mu)
    
    
    return next_values

# try sample from all gaussians and see the difference?

def max_sample(pi, sigma, mu):
    #  CHANGE: Instead of random sampling, use the mean of the most probable component
    alpha_idx = torch.argmax(pi, dim=2)  # Find the index of the most probable Gaussian component
    selected_mu = mu.gather(2, alpha_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1,-1, mu.size(-1)))
    next_values = selected_mu[:, -1, :]  # Use the mean of the selected component
    return next_values

def select_sample(pi, sigma, mu, selected_gaussian_idx=4, variance_div=1000):
    """
    Samples from the specified Gaussian component of the mixture.

    :param pi: Mixture component weights [B, T, G]
    :param sigma: Standard deviations of mixture components [B, T, G, O]
    :param mu: Means of mixture components [B, T, G, O]
    :param selected_gaussian_idx: Index of the Gaussian component to sample from [B, T]
    :param variance_div: Factor to divide the variance for controlling sample spread
    :return: Sampled values from the specified Gaussian component of the MDN
    """
    # Select the specified Gaussian components
    
    # Convert selected_gaussian_idx to a tensor if it is not already
    if not isinstance(selected_gaussian_idx, torch.Tensor):
        B, T, _ = pi.shape  # Get the dimensions from the shape of pi
        selected_gaussian_idx = torch.full((B, T), selected_gaussian_idx, dtype=torch.long, device=pi.device)

    selected_mu = torch.gather(mu, 2, selected_gaussian_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, mu.size(-1)))
    selected_sigma = torch.gather(sigma, 2, selected_gaussian_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, sigma.size(-1)))
    selected_sigma = selected_sigma / variance_div

    # Create the normal distribution and sample from it
    normal_dist = Normal(selected_mu, selected_sigma)
    next_values = normal_dist.sample()[:, -1, :]  # Sample from the selected Gaussian

    return next_values

def calculate_dynamic_emotion_scores(mu, sigma, emotion_logits, k, emotion_weight, neutral_index=4):
    """
    Calculate scores for each Gaussian component based on dynamic movement, noise, and emotion.
    Emphasize movements that are furthest from 'Neutral'.

    :param mu: Means of Gaussian components [B, T, G, O]
    :param sigma: Standard deviations of Gaussian components [B, T, G, O]
    :param emotion_logits: Emotion logits [B, Emotion_Categories]
    :param k: Weight for the noise component
    :param emotion_weight: Weight for the emotion component
    :param neutral_index: Index of 'Neutral' in emotion logits
    :return: A tensor of scores [B, T, G]
    """
    movement_score = torch.sqrt(torch.sum(mu**2, dim=3))
    noise_penalty = k * torch.sqrt(torch.sum(sigma**2, dim=3))

    # Get the emotion score, penalizing 'Neutral' emotions
    neutral_score = emotion_logits[:, neutral_index].unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
    non_neutral_bonus = 1 - neutral_score  # Bonus for being non-neutral
    dominant_emotion = emotion_logits.max(dim=1).values.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
    emotion_score = dominant_emotion * non_neutral_bonus  # Emphasize non-neutral emotions

    scores = movement_score - noise_penalty + emotion_weight * emotion_score
    
    print("movement score: ", movement_score)

    return scores


def sample_dynamic_emotion(pi, sigma, mu, emotion_logits, k=1.0, emotion_weight=1.0):
    """
    Sample from the MDN focusing on dynamic movement, less noise, and emotional relevance.

    :param pi: Mixture component weights [B, T, G]
    :param sigma: Standard deviations of mixture components [B, T, G, O]
    :param mu: Means of mixture components [B, T, G, O]
    :param emotion_logits: Emotion logits [B, T, Emotion_Categories]
    :param k: Weight for the noise component in scoring
    :param emotion_weight: Weight for the emotion component in scoring
    :return: Sampled values from the MDN
    """
    # Calculate scores for each Gaussian component
    scores = calculate_dynamic_emotion_scores(mu, sigma, emotion_logits, k, emotion_weight)

    # Select the Gaussian component with the highest score
    selected_gaussian_idx = torch.argmax(scores, dim=2)

    # Sample from the selected Gaussian components
    selected_samples = select_sample(pi, sigma, mu, selected_gaussian_idx)

    return selected_samples

def calculate_dynamic_emotion_scores_individual(mu, sigma, emotion_logits, k, emotion_weight, neutral_index=4):
    """
    Calculate scores for each keypoint of each Gaussian component based on dynamic movement, noise, and emotion.

    :param mu: Means of Gaussian components [B, T, G, O]
    :param sigma: Standard deviations of Gaussian components [B, T, G, O]
    :param emotion_logits: Emotion logits [B, Emotion_Categories]
    :param k: Weight for the noise component
    :param emotion_weight: Weight for the emotion component
    :param neutral_index: Index of 'Neutral' in emotion logits
    :return: A tensor of scores [B, T, G, O]
    """
    B, T, G, O = mu.shape

    # Movement score (dynamic movement)
    movement_score = torch.sqrt(torch.sum(mu**2, dim=3), )  # [B, T, G]

    # Noise penalty
    noise_penalty = k * torch.sqrt(torch.sum(sigma**2, dim=3))  # [B, T, G]

    # Emotion score (including penalizing 'Neutral' emotions)
    neutral_score = emotion_logits[:, neutral_index].unsqueeze(1).unsqueeze(2).expand(-1, T, G)  # [B, T, G]
    non_neutral_bonus = 1 - neutral_score
    dominant_emotion = emotion_logits.max(dim=1).values.unsqueeze(1).unsqueeze(2).expand(-1, T, G)  # [B, T, G]
    emotion_score = dominant_emotion * non_neutral_bonus

    # Expand the scores to have the same shape as mu and sigma for each keypoint
    expanded_movement_score = movement_score.unsqueeze(-1).expand(-1, -1, -1, O)  # [B, T, G, O]
    expanded_noise_penalty = noise_penalty.unsqueeze(-1).expand(-1, -1, -1, O)  # [B, T, G, O]
    expanded_emotion_score = emotion_score.unsqueeze(-1).expand(-1, -1, -1, O)  # [B, T, G, O]

    # Final scores for each keypoint
    scores = expanded_movement_score - expanded_noise_penalty + emotion_weight * expanded_emotion_score

    return scores


def sample_dynamic_emotion_individual(pi, sigma, mu, emotion_logits, k=2.0, emotion_weight=2.0, variance_div=1000):
    """
    Sample from the MDN focusing on individual keypoints with emotion awareness.

    :param pi: Mixture component weights [B, T, G]
    :param sigma: Standard deviations of mixture components [B, T, G, O]
    :param mu: Means of mixture components [B, T, G, O]
    :param emotion_logits: Emotion logits [B, Emotion_Categories]
    :param k: Weight for the noise component in scoring
    :param emotion_weight: Weight for the emotion component in scoring
    :return: Sampled values [B, T, O]
    """
    B, T, G, O = mu.shape
    selected_samples = torch.zeros((B, T, O))

    # Assume a function calculate_dynamic_emotion_scores_individual returns [B, T, G, O]
    scores = calculate_dynamic_emotion_scores_individual(mu, sigma, emotion_logits, k, emotion_weight)
    
    for o in range(O):
        selected_gaussian_idx_o = torch.argmax(scores[:, :, :, o], dim=2)  # [B, T]
        idx_o = selected_gaussian_idx_o.unsqueeze(-1)  # [B, T, 1]
        
        # Sample from the selected Gaussian component for keypoint 'o'
        selected_mu_o = torch.gather(mu[:, :, :, o], 2, idx_o).squeeze(2)  # [B, T]
        selected_sigma_o = torch.gather(sigma[:, :, :, o], 2, idx_o).squeeze(2) / variance_div  # Adjust variance if needed

        normal_dist = Normal(selected_mu_o, selected_sigma_o)
        selected_samples[:, :, o] = normal_dist.sample()

    return selected_samples.to(device)

