
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


class EmotionFC2(nn.Module):
    def __init__(self, in_features, out_features):
        super(EmotionFC2, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        return self.activation(self.fc(x))





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



def sample(last_frames, pi, sigma, mu , variance_div= 100):
    B, T, G, O = mu.shape
    
    # CHANGE: Instead of random sampling, use the mean of the most probable component
    # alpha_idx = torch.argsort(pi, dim=2, descending=False)  # Find the index of the most probable Gaussian component
    # alpha_idx = adjust_movement_rankings(alpha_idx, valid_ranks=[1])  # Find the index of the second most probable Gaussian component
    # alpha_idx = torch.argmax(alpha_idx, dim=2)
    
    # get gaussian with max movement
    # Add the last frame of last_frames to the beginning of mu for delta calculation
    extended_mu = torch.cat([last_frames[:, -1, :].unsqueeze(1).expand(-1, G, -1).unsqueeze(1), mu], dim=1)

    # Calculate deltas (difference in position) for each keypoint across each frame
    deltas = extended_mu[:, 1:, :, :] - extended_mu[:, :-1, :, :]  # [B, T, G, O]
    # Movement score for each keypoint
    movement_score = torch.sqrt(torch.sum(deltas**2, dim=3))  # [B, T, G]
    # Gather mu and sigma based on the ranked indices
    ranked_movement = torch.argsort(movement_score,dim=2, descending=True)
    alpha_idx = adjust_movement_rankings(ranked_movement, valid_ranks=[4])  # Find the index of the  max movement Gaussian component
    alpha_idx = torch.argmax(alpha_idx, dim=2)
    
    # get mu of most probable gaussian
    max_alpha_idx = torch.argmax(pi, dim=2)
    
    # gather both
    most_prob_mu = mu.gather(2, alpha_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1,-1, mu.size(-1)))
    max_mu = mu.gather(2, max_alpha_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1,-1, mu.size(-1)))
    
    movement_weight = 0.5
    # find the midpoint between the two
    selected_mu = torch.lerp(most_prob_mu, max_mu, movement_weight)
    selected_sigma = sigma.gather(2, alpha_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, sigma.size(-1)))
    selected_sigma = selected_sigma/variance_div
    
    # print(variance_div)
    # Divide by 100 to reduce the variance of the selected Gaussian I think, Pette et al 2019 did this not sure why
    # but it seems to help model be less jaggy? - sigma is variance, so smaller sigma is closer to mean - less jaggy but more boring
    
    normal_dist = Normal(selected_mu, selected_sigma)
    next_values = normal_dist.sample()[:, -1, :]  # Sample from the selected Gaussian
    
    # random sample - previous implementation
    # next_values = mdn.sample(pi, sigma, mu)
    
    
    return next_values

# try sample from all gaussians and see the difference?
def adjust_movement_rankings(ranked_movement, valid_ranks=[1]):
    """
    Adjust movement rankings to only consider specific ranks.
    
    :param ranked_movement: The tensor of ranked movement scores [B, T, G].
    :param valid_ranks: The ranks to consider (e.g., [1, 2]).
    :return: Adjusted rankings where non-valid ranks are set to -1.
    """
    B, T, G = ranked_movement.shape
    # Initialize a mask with False values
    valid_mask = torch.zeros_like(ranked_movement, dtype=torch.bool)

    # Set True for valid ranks
    for rank in valid_ranks:
        valid_mask |= (ranked_movement == rank)

    # Adjust rankings to set non-valid ranks to -1
    adjusted_rankings = torch.where(valid_mask, ranked_movement, torch.tensor(-1, device=ranked_movement.device))

    return adjusted_rankings


def ranked_scores(last_frames,  pi,sigma, mu):
    """
    rank pi, mu and sigma, shape [B, T, G]
    for mu and sigma, all O are ranked together by average of all O - This is fine because pi rank is shared, so it's consistent
    pi is smoothness
    """
    B, T, G, O = mu.shape

    # Get the indices that would sort pi in descending order, so highest probabilities come first
    ranked_indices = torch.argsort(pi, dim=2, descending=True)   # [B, T, G,O], sort according to g

    # Add the last frame of last_frames to the beginning of mu for delta calculation
    extended_mu = torch.cat([last_frames[:, -1, :].unsqueeze(1).expand(-1, G, -1).unsqueeze(1), mu], dim=1)

    # Calculate deltas (difference in position) for each keypoint across each frame
    deltas = extended_mu[:, 1:, :, :] - extended_mu[:, :-1, :, :]  # [B, T, G, O]
    

    # Movement score for each keypoint
    movement_score = torch.sqrt(torch.sum(deltas**2, dim=3))  # [B, T, G]

    # Noise penalty for each keypoint
    noise_penalty = torch.sqrt(torch.sum(sigma**2, dim=3))  # [B, T, G]

    # Gather mu and sigma based on the ranked indices
    ranked_movement = torch.argsort(movement_score,dim=2, descending=True)
    ranked_movement = adjust_movement_rankings(ranked_movement)
    ranked_noise = torch.argsort(noise_penalty,dim=2, descending=False)
    
    scores = ranked_indices + ranked_movement + ranked_noise
    
    max_scores  = torch.argmax(scores, dim=2)

    return max_scores


def select_and_sample_gaussians(last_frames,pi,sigma,mu, emotion_fc2, attention_pooling, input_emotion_logits, variance_div=100):
    B, T, G, O = mu.shape
    device = mu.device

    # Initialize tensors to hold the final selected samples
    selected_samples = torch.zeros((B, O), device=device)
    closest_distance = torch.full((B,), float('inf'), device=device)
    
    
    alpha_idx = ranked_scores(last_frames,pi,sigma,mu) # Find the index of the most probable Gaussian component
    chosen_mu = mu.gather(2, alpha_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1,-1, mu.size(-1)))
    chosen_sigma = sigma.gather(2, alpha_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, sigma.size(-1)))
    
    # Remove the extra G dimension since we have selected the component
    chosen_sigma = chosen_sigma.squeeze(2)/variance_div
    chosen_mu = chosen_mu.squeeze(2)
    
    # Sample from the normal distributions
    normal = torch.distributions.Normal(chosen_mu, chosen_sigma)
    
    selected_sample = normal.sample()  # [B, T, O]
    
    # # sample 3 times, get the one that is closest to the input emotion
    # for i in range(3):
    #     sample = normal.sample()
    #     pooled_sample = attention_pooling(sample)
    #     emotion_logits_sample = emotion_fc2(pooled_sample)
    #     distance = emotion_distance(emotion_logits_sample, input_emotion_logits)
    #     closest_distance = torch.min(closest_distance, distance)
    #     mask = distance < closest_distance
    #     closest_distance[mask] = distance[mask]
    #     selected_sample[mask] = sample[mask]
    
            
    return selected_sample[:, -1, :]




def emotion_distance(emotion_logits_gaussians, input_emotion_logits):
    """
    Calculate the distance between the emotion logits of Gaussian components and the input emotion logits.

    :param emotion_logits_gaussians: Emotion logits for each Gaussian [B, G, Emotion_Categories]
    :param input_emotion_logits: Input emotion logits [B, Emotion_Categories]
    :return: Distance for each Gaussian component [B, G]
    """
    # Calculate Euclidean distance or another distance metric
    distance = torch.norm(emotion_logits_gaussians - input_emotion_logits, dim=1)
    return distance

# def select_closest_gaussian(mu, sigma, pi, emotion_fc2, attention_pooling, input_emotion_logits, variance_div=100):
#     """
#     Select the Gaussian component closest to the input emotion for sampling.
#     """
#     B, T, G, O = mu.shape
#     device = mu.device
#     selected_samples = torch.zeros((B, O), device=device)
    
#     # Initialize tensor to store pooled emotion logits for each Gaussian
#     # Correct initialization assuming 5 Gaussians, 7 emotion categories, batch size 8
#     pooled_emotion_logits = torch.zeros((B, G, 7), device=device)

    
#     for g in range(G):
#         # Extract and reshape mu for the g-th Gaussian to apply attention pooling: [B, T, O] -> [B, T, O]
#         mu_g = mu[:, :, g, :]
        
#         # Apply attention pooling to mu_g: [B, T, O] -> [B, O]
#         pooled_mu_g = attention_pooling(mu_g)
        
#         # Convert pooled motion features to emotion logits: [B, O] -> [B, Emotion_Categories]
#         emotion_logits_g = emotion_fc2(pooled_mu_g)
        
#         # Store the pooled emotion logits
#         pooled_emotion_logits[:, g, :] = emotion_logits_g
    
#     # Calculate distance between pooled emotion logits of each Gaussian and the input emotion logits: [B, G, O]
#     distances = torch.norm(pooled_emotion_logits - input_emotion_logits.unsqueeze(1), dim=2)
    
#     # Find the index of the closest Gaussian component for each batch item: [B]
#     closest_idxs = torch.argmin(distances, dim=1)
    
#     # Sample from the selected Gaussian for each batch item
#     for b in range(B):
#         idx = closest_idxs[b]
#         selected_mu = mu[b, :, idx, :].mean(dim=0)  # [O]
#         selected_sigma = sigma[b, :, idx, :].mean(dim=0) / variance_div  # [O]
        
#         # Sample from the Normal distribution defined by the selected Gaussian's parameters
#         normal_dist = Normal(selected_mu, selected_sigma)
#         selected_samples[b, :] = normal_dist.sample()

#     return selected_samples