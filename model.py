# Prototype 8 
# set up environment
import glob
import os 
import numpy as np
import torch
import json
from tqdm import tqdm
import logging
from typing import List
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import matplotlib.pyplot as plt
import random
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import *
import argparse
import sys
import libs.mdn as mdn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger()
# Clear previous handlers
for handler in logger.handlers[:]:
    handler.close()
    logger.removeHandler(handler)

logging.basicConfig(filename= f"logs/preprocessing_log.txt", level=logging.INFO, filemode='w')
# logging clear file
logger = logging.getLogger()

# Set root directory
root_dir = "C:\\Users\\avika\\OneDrive\\Documents\\UAL\\interactive_dance_thesis"
os.chdir(root_dir)

# Check if the current working directory was set correctly
print(os.getcwd())



# Functions--------------------------------------------------


def get_batch(split, block_size, batch_size, train_data, train_emotions, val_data, val_emotions, device=device):
    data, emotions = (train_data, train_emotions) if split == 'train' else (val_data, val_emotions)
    
    # Filter out videos that are too short - blocksize+1
    valid_indices = [i for i, video in enumerate(data) if len(video) > block_size+1]
    
    # If there are not enough valid videos, throw an error
    if len(valid_indices) < batch_size:
        raise ValueError("Not enough videos longer than block_size. Reduce block_size or use more/longer videos.")
    
    # Choose random videos
    selected_indices = random.sample(valid_indices, batch_size)

    # For each chosen video, select a random starting point
    start_frames = [random.randint(0, len(data[i]) - (block_size +1)) for i in selected_indices]

    # Extract subsequences from each chosen video and convert to tensors
    x = torch.stack([torch.tensor(data[i][start:start + block_size], dtype=torch.float32) for i, start in zip(selected_indices, start_frames)])
    y = torch.stack([torch.tensor(data[i][start + 1:start + block_size + 1], dtype=torch.float32) for i, start in zip(selected_indices, start_frames)])
    
    # Extract the corresponding emotion vectors
    e = torch.stack([torch.tensor(emotions[i], dtype=torch.float32) for i in selected_indices])

    # Compute the mask to mask out -inf values
    mask = (x != float('-inf')).all(dim=-1).float()  # assuming -inf is present in any part of the data point

    # Move tensors to the designated device
    x, y, e, mask = x.to(device), y.to(device), e.to(device), mask.to(device)
    
    return x, y, e, mask


# Decoder Model------------------------------------


def positional_encoding(seq_len, d_model):
    """
    Returns the positional encoding for a given sequence length and model size.

    Parameters:
    - seq_len (int): Length of the sequence.
    - d_model (int): Size of the model embedding.

    Returns:
    - A tensor of shape (seq_len, d_model) containing the positional encoding.
    """
    
    position = torch.arange(seq_len).unsqueeze(1).float() # [seq_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                         (-math.log(10000.0) / d_model))  # [d_model/2]
    pos_enc = torch.zeros((seq_len, d_model))

    pos_enc[:, 0::2] = torch.sin(position * div_term) # apply sin to even indices in the array; 2i
    pos_enc[:, 1::2] = torch.cos(position * div_term) # apply cos to odd indices in the array; 2i+1

    return pos_enc


class Head(nn.Module):
    """one head of self attention"""
    
    def __init__(self,head_size,n_emb,dropout=0.2,blocksize=16):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False, device=device)
        self.query = nn.Linear(n_emb, head_size, bias=False, device=device)
        self.value = nn.Linear(n_emb, head_size, bias=False, device=device)
        self.register_buffer('tril', torch.tril(torch.ones(blocksize, blocksize)))
        self.n_emb = n_emb
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, emotion_features=None):
        B,T,C = x.shape # batch size, time, context
        
        if emotion_features is not None:
            # concatenate emotion features to the queries
            emotion_features_expanded = emotion_features.unsqueeze(1).expand(-1, T, -1)
            q = self.query(torch.cat((x, emotion_features_expanded), dim=-1))
        else:
            q = self.query(x)
        
        # key, query, value
        k = self.key(x) # B,T,C
        v= self.value(x) # B,T,C
        
        # compute attention scores ("affinities")
         # Scaled dot-product attention - same as below
        # attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / math.sqrt(self.embed_size)

        wei = q @ k.transpose(-1,-2) # B,T,T
        wei /= math.sqrt(self.n_emb) # scale by sqrt of embedding dimension
        self.tril = self.tril.to(device)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # mask out upper triangular part so don't attend to future
        wei = F.softmax(wei, dim=-1) # B,T,T
        wei = self.dropout(wei)
        # apply attention to values - weighted aggregation
        out = wei @ v # (B,T,T) @ (B,T,C) --> B,T,C
        
        return out
        
        
class MultiHeadAttention(nn.Module):
    
    def __init__(self,num_heads,head_size,n_emb,dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size,n_emb) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb, bias=False, device=device) # (B,T,C) - projection back into residual pathway
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, emotion_features=None):
        # x is (B,T,C)
        out = torch.cat([h(x, emotion_features) for h in self.heads], dim=-1) # (B,T,C*num_heads)
        out = self.dropout(self.proj(out)) # (B,T,C) - projection back into residual pathway
        
        return out
    
class FeedForward(nn.Module):
    """A simple lineear layer followed by a ReLU - allows all tokens to think on data individually"""
    
    def __init__(self,n_emb,dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb , device=device), # 4 * because recommended in paper residual pathway - growing residual pathway
            nn.LeakyReLU(0.2, inplace=True), #leaky relu instead of relu because Pettee et al 2019 - alpha 0.2
            nn.Linear( 4* n_emb, n_emb , device=device), # required otherwise output will collapse  - projection back into residual pathway
            nn.Dropout(dropout)
          
        )
    
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """Transformer Block: communication followed by computation - basically self attention heads and feedforward"""

    def __init__(self, n_emb, n_heads,dropout=0.2):
        
        super().__init__()
        head_size = n_emb//n_heads
        self.sa = MultiHeadAttention(num_heads=n_heads, head_size=head_size, n_emb=n_emb)
        self.ffwd = FeedForward(n_emb=n_emb)
        # self.ln1 =  nn.InstanceNorm1d(n_emb , device=device)
        # self.ln2 =  nn.InstanceNorm1d(n_emb, device=device)
        # 2* because concatenate emotion and keypoints
        self.ln1 =  nn.LayerNorm(n_emb , device=device)
        self.ln2 =  nn.LayerNorm(n_emb, device=device)
        self.hidden_dim = n_emb

        
    def forward(self, x):
        # x + due to residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
   
class MotionModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, emotion_dim=7, blocksize = 16, hidden_dim=256, n_layers=8 , dropout=0.2, device = device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False, device=device) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False,device=device)
        self.mdn = mdn.MDN(output_dim,output_dim, num_gaussians=5) # fine
        # emotions
        self.emotion_fc = nn.Linear(emotion_dim, hidden_dim, bias=False,device=device)
        self.emotion_dropout = nn.Dropout(dropout)
        self.emotion_head = nn.Linear(hidden_dim, emotion_dim).to(device)  # Outputs a continuous vector for emotions
        
        self.positional_encoding = positional_encoding(seq_len=blocksize, d_model=hidden_dim).to(device)
        layers = [Block(n_emb=hidden_dim, n_heads=4) for _ in range(n_layers)]
        layers.append(nn.LayerNorm(hidden_dim, device=device))
        # layers.append(nn.InstanceNorm1d(hidden_dim, device=device))
        self.blocks = nn.Sequential(*layers)

        self.lm_head = nn.Linear(hidden_dim, hidden_dim, bias=False, device=device)
       
    
        
    def forward(self, inputs,  targets=None , emotions =None, l1_lambda = 0.001, mask=None,USE_MDN=True, threshold=0.1, PENALTY=False):
        B,T,C = inputs.shape # batch size, time, context
        
        # fc1 transforms input into hidden dimension
        keypoint_features = self.fc1(inputs) # B,T,hidden dimension
        # Add positional encoding
        keypoint_features += positional_encoding(seq_len=T, d_model=self.hidden_dim).to(device) # positional_encoding = T,hidden dimension , added = B,T,hidden dimension
        
        # Process emotion inputs
        emotion_features = self.emotion_fc(emotions)  # B, emotion_dim to B, hidden_dim
        emotion_features = self.emotion_dropout(emotion_features)
        emotion_features = emotion_features.unsqueeze(1).expand(-1, T, -1)  # B, T, hidden_dim
        
        
        # FIGURE OUT MULTIMODAL INPUT TORCH
        # Combine keypoint and emotion features
        # x = torch.cat((keypoint_features, emotion_features), dim=-1)  # Concatenate along the feature dimension
        x = keypoint_features + emotion_features
        x = self.blocks(x) # B,T,hidden dimension
        
        # Deconcatenate keypoint and emotion features
        # keypoint_features, emotion_features = x.split([self.hidden_dim, self.hidden_dim], dim=-1)
        # keypoint_features, emotion_features = x

        # Save the latent vectors
        latent_vectors = x.detach()
        
        x= self.lm_head(x) # B,T,hidden dimension
        
        # fc2 transforms hidden dimension into output dimension
        logits = self.fc2(x)
        
        # Output emotion logits - emotions which was used to condition the model 
        emotion_logits = self.emotion_head(torch.mean(x, dim=1) )  # B, emotion_dim
        
        if USE_MDN:
        # Apply MDN after dense layer  - look at https://github.com/deep-dance/core/blob/27e9c555d1c85599eba835d59a79cabb99b517c0/creator/src/model.py#L59
            pi, sigma, mu = self.mdn(logits) #fine
        
        if targets is None:
            loss = None
           
        
        else:
            B,T,C = inputs.shape # batch size, time, context
            # You can adjust this value based on your needs
           
            if L1_LAMBDA is None:
                
                # if penalising model for small movements
                if PENALTY:
                    
                    # Current MSE loss calculation
                    mse_loss = F.mse_loss(logits, targets)

                    # Extract the magnitude of the deltas from logits
                    logits_deltas_magnitude = torch.abs(logits[:, :, 50:100])

                    # Calculate the cumulative magnitude of the deltas over the entire sequence for each sample in the batch
                    cumulative_deltas_magnitude = logits_deltas_magnitude.sum(dim=1)

                    # Calculate the penalty for samples where the cumulative magnitude of deltas is below the threshold
                    penalty_mask = (cumulative_deltas_magnitude < threshold).float()
                    penalty = penalty_mask.mean()

                    # Hyperparameter to balance the original MSE and the new penalty term
                    alpha = 0.01  # This value can be adjusted based on your needs

                    # Combine the MSE loss and the penalty term to get the modified loss
                    loss = mse_loss + alpha * penalty
                
                else:
                    if USE_MDN:
                        # loss =  mdn.mdn_loss(pi, sigma, mu, targets)+ F.mse_loss(emotion_logits, emotions)
                        # loss = F.mse_loss(logits, targets) + (F.mse_loss(emotion_logits, emotions)) + mdn.mdn_loss(pi, sigma, mu, targets)
                        loss = (F.mse_loss(logits, targets) , (F.mse_loss(emotion_logits, emotions)) , mdn.mdn_loss(pi, sigma, mu, targets))
                        
                    else:
                        # loss = F.mse_loss(logits, targets) + (F.mse_loss(emotion_logits, emotions))
                        loss = (F.mse_loss(logits, targets) , (F.mse_loss(emotion_logits, emotions)))
                
 
            else:
                l1_norm = sum(p.abs().sum() for p in m.parameters())  # Calculate L1 norm for all model parameters
                loss = F.mse_loss(logits, targets) + l1_lambda * l1_norm

           
        if USE_MDN:
            return pi, sigma, mu, logits, emotion_logits, loss,latent_vectors
        else:
            return logits,emotion_logits,loss,latent_vectors
    
    def generate(self, inputs, emotions, max_new_tokens, block_size=16, USE_MDN=True):
        generated_sequence = inputs
        generated_emotions = emotions

        for _ in range(max_new_tokens):
            cond_sequence = generated_sequence[:, -block_size:]  # Use the block_size parameter

            if USE_MDN:
                pi, sigma, mu, logits, emotion_logits, _, _ = self(inputs=cond_sequence, emotions=generated_emotions)

                # CHANGE: Instead of random sampling, use the mean of the most probable component
                # next_values = mdn.max_sample(pi, sigma, mu)
                next_values = mdn.sample(pi, sigma, mu)
                
                # random sample - previous implementation
                # next_values = mdn.sample(pi, sigma, mu)

                generated_sequence = torch.cat([generated_sequence, next_values], dim=1)

            else:
                logits, emotion_logits, _, _ = self(inputs=cond_sequence, emotions=generated_emotions)
                next_values = logits[:, -1, :]  # get the last token from the logits
                generated_sequence = torch.cat([generated_sequence, next_values.unsqueeze(1)], dim=1)

            generated_emotions = emotion_logits

        return generated_sequence, generated_emotions

    
# train----------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    eval_iters = 100
    print('Evaluating loss...')
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters), desc=f"Evaluating Loss", unit="batch"):
            xb, yb, eb, _ = get_batch(split, BLOCK_SIZE, BATCH_SIZE, train_data,train_emotions, val_data, val_emotions)
            if USE_MDN:
                _,_,_,_,_, loss,_ = m(xb, yb, eb)
                mse_logits_loss, mse_emotion_loss, mdn_loss = loss
                total_loss = mse_logits_loss + mse_emotion_loss + mdn_loss
            else:
                _,_, loss,_ = m(xb, yb, eb)
                mse_logits_loss, mse_emotion_loss = loss
                total_loss = mse_logits_loss + mse_emotion_loss
            losses[k] = total_loss.item()
        out[split] = losses.mean()
    m.train()
    return out


# test----------------------------------------------------
def unnormalise_list_2D(data_tensor, max_x, min_x, max_y, min_y, max_dx, min_dx, max_dy, min_dy):
    all_frames = []
    
    # Loop through each batch
    for batch_idx in range(data_tensor.size(0)):
        batch_frames = []
        
        # Loop through each frame in the batch
        for frame_idx in range(data_tensor.size(1)):
            frame_data = data_tensor[batch_idx, frame_idx, :]
            unnormalized_data = []
            
            # Unnormalize the first 50 values (absolute x and y coordinates)
            for i in range(0, 50, 2):
                x = frame_data[i]
                y = frame_data[i+1]
                unnormalized_x = (x + 1) / 2 * (max_x - min_x) + min_x
                unnormalized_y = (y + 1) / 2 * (max_y - min_y) + min_y
                unnormalized_data.extend([unnormalized_x.item(), unnormalized_y.item()])
            
            # Unnormalize the second 50 values (x and y deltas)
            for i in range(50, 100, 2):
                dx = frame_data[i]
                dy = frame_data[i+1]
                unnormalized_dx = (dx + 1) / 2 * (max_dx - min_dx) + min_dx
                unnormalized_dy = (dy + 1) / 2 * (max_dy - min_dy) + min_dy
                unnormalized_data.extend([unnormalized_dx.item(), unnormalized_dy.item()])
            
            # Append the emotion encoding without unnormalizing
            unnormalized_data.extend(frame_data[-7:].tolist())
            batch_frames.append(unnormalized_data)
        all_frames.append(batch_frames)
    return all_frames

def plot_losses(train_losses, val_losses, EPOCHS, spacing,train_seed, max_ticks=10):
    plt.figure(figsize=(12,6))
    
    # Calculate x-axis values for the epochs based on the original spacing
    x_values = list(range(spacing, spacing * len(train_losses) + 1, spacing))
    
    # Dynamically determine tick spacing based on total epochs and a maximum number of ticks
    # Ensuring that tick_spacing is a multiple of the provided spacing
    tick_spacing = max(spacing, (EPOCHS // max_ticks) // spacing * spacing)
    
    # Calculate x-axis tick values and labels based on dynamic tick spacing
    x_ticks = list(range(tick_spacing, EPOCHS + 1, tick_spacing))
    x_labels = [str(i) for i in x_ticks]
    
    plt.plot(x_values, train_losses, label='Training Loss')  # Use x_values for x-values
    plt.plot(x_values, val_losses, label='Validation Loss')  # Use x_values for x-values
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(x_ticks, x_labels)  # Set x-axis ticks and labels based on dynamic tick spacing
    plt.legend(loc='upper right')  # Explicitly specify the legend location
    plt.title(f'Training and Validation Loss')
    
    plt.tight_layout()  # Ensure elements fit within the figure
    
    # Use the train seed in the filename
    plot_path = os.path.join("D:/Interactive Dance Thesis Tests/TransformerResults/losses", f"loss_plot_{train_seed}.png")
    
    # Save the plot
    plt.savefig(plot_path)
    plt.close()
    return f"Plot saved to {plot_path}"

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_path):
    """Save the model checkpoint."""
    # Use the run seed in the filename
    # checkpoint_path = os.path.join(checkpoint_dir, f"MEED_checkpoint_{run_seed}.pth")
    
    
    # if not os.path.exists(checkpoint_path):
    #     print('Creating checkpoints directory...')
    #     os.makedirs(checkpoint_path)
    
    print(f"Saving model checkpoint to {checkpoint_path}")
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(),
             'epoch': epoch,
             'loss': loss,
             'train_seed' : train_seed}
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer,checkpoint_path, scheduler = None):
    """Load the model checkpoint. - if no scheduler, pass None"""
    print('Loading checkpoint...')
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None: #  for backwards compatability
        scheduler.load_state_dict(state['scheduler'])
    epoch = state['epoch']
    loss = state['loss']
    train_seed = state['train_seed']
    print(f"Checkpoint loaded from {checkpoint_path}")
    return model, optimizer, scheduler, epoch, loss,train_seed

def visualise_skeleton(all_frames, max_x, max_y,emotion_vectors=None, max_frames=500, save=False, save_path=None, prefix=None, train_seed=None , delta=False, destroy = True):
    """Input all frames dim 50xn n being the number of frames 50= 25 keypoints x and y coordinates"""

    # visualise to check if the data is correct
    # BODY_25 Keypoints
    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                        'L-Elb', 'L-Wr', 'MidHip', 'R-Hip', 'R-Knee', 'R-Ank', 
                        'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 
                        'L-Ear', 'L-BigToe', 'L-SmallToe', 'L-Heel', 'R-BigToe', 
                        'R-SmallToe', 'R-Heel']


    limb_connections = [
        ("Nose", "Neck"),
        ("Neck", "R-Sho"),
        ("R-Sho", "R-Elb"),
        ("R-Elb", "R-Wr"),
        ("Neck", "L-Sho"),
        ("L-Sho", "L-Elb"),
        ("L-Elb", "L-Wr"),
        ("Neck", "MidHip"),
        ("MidHip", "R-Hip"),
        ("R-Hip", "R-Knee"),
        ("R-Knee", "R-Ank"),
        ("MidHip", "L-Hip"),
        ("L-Hip", "L-Knee"),
        ("L-Knee", "L-Ank"),
        ("Nose", "R-Eye"),
        ("R-Eye", "R-Ear"),
        ("Nose", "L-Eye"),
        ("L-Eye", "L-Ear"),
        ("L-Ank", "L-BigToe"),
        ("L-Ank", "L-SmallToe"),
        ("L-Ank", "L-Heel"),
        ("R-Ank", "R-BigToe"),
        ("R-Ank", "R-SmallToe"),
        ("R-Ank", "R-Heel")
    ]
    
     # Define a mapping from emotion vectors to emotion labels
    # Define emotion labels
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sad', 'Surprise']
    
    # Initialize a blank canvas (image)
    canvas_size = (int(max_y)+50, int(max_x)+50, 3)  
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    
    
    if save:
        # Determine the save path
        if save_path is None:
            save_path = f"D:\\Interactive Dance Thesis Tests\\TransformerResults\\{train_seed}"

        # Ensure directory exists
        if not os.path.exists(save_path):
            print(f"Creating directory {save_path}")
            os.makedirs(save_path)

        # Determine a unique filename
        existing_files = os.listdir(save_path)
        file_num = 1
        while f"{prefix or ''}{file_num}.mp4" in existing_files:
            file_num += 1
        out_path = os.path.join(save_path, f"{prefix or ''}{file_num}.mp4")

        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, 10.0, (canvas_size[1], canvas_size[0]))
    
    previous_frame_data = None
    
    # Iterate over all frames; the first frame uses absolute keypoints, the rest use relative keypoints (deltas)
    for frame_data in tqdm(all_frames[:max_frames], desc="Visualizing frames"):
        
        # If previous_frame_data is None, this is the first frame and we use absolute positions.
        # Otherwise, add the delta to the previous frame's keypoints to get the new keypoints
        if delta ==True:
            if previous_frame_data is not None:
                frame_data[:50] = [prev + delta for prev, delta in zip(previous_frame_data[:50], frame_data[50:100])]
        
            # Update previous_frame_data
            previous_frame_data = copy.deepcopy(frame_data)
        
        canvas_copy = canvas.copy()
        
        # Extract x, y coordinates and emotion vector
        x_coords = frame_data[0:50:2] 
        y_coords = frame_data[1:50:2]
        
        
        xy_coords = list(zip(x_coords, y_coords))
        sane = sanity_check(xy_coords)
        # Plot keypoints on the canvas
        for i, (x, y) in enumerate(xy_coords):
            if sane[i] == False:
                continue
            x_val = x.item() if torch.is_tensor(x) else x
            y_val = y.item() if torch.is_tensor(y) else y
            cv2.circle(canvas_copy, (int(x_val), int(y_val)), 3, (0, 0, 255), -1)  
            cv2.putText(canvas_copy, keypointsMapping[i], (int(x_val), int(y_val)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw connections (limbs) on the canvas
        for limb in limb_connections:
            start_idx = keypointsMapping.index(limb[0])
            end_idx = keypointsMapping.index(limb[1])
            
            start_point = (int(x_coords[start_idx]), int(y_coords[start_idx]))
            end_point = (int(x_coords[end_idx]), int(y_coords[end_idx]))

            if start_point == (0,0) or end_point == (0,0) or not sane[start_idx] or not sane[end_idx]:
                continue
            cv2.line(canvas_copy, start_point, end_point, (0, 255, 0), 2)  
        
        # Display the emotion percentages and labels on the top right of the frame
        
       # Assuming emotion_vectors is a tuple (emotion_in, emotion_out)
        emotion_in, emotion_out = emotion_vectors

        # Calculate percentages for emotion_in
       
        emotion_in_percentages = [
            f"{int(e * 100)}% {emotion_labels[i]}" 
            for i, e in enumerate(emotion_in.tolist()) if round(e * 100) > 0
        ]


        # Calculate percentages for emotion_out
        
        emotion_out_percentages = [
            f"{int(e * 100)}% {emotion_labels[i]}" 
            for i, e in enumerate(emotion_out.tolist()) if round(e * 100) > 0
        ]

        y0, dy = 30, 15  # Starting y position and line gap

        # Label positions
        label_in_position = (10, y0 - dy)  # Position for 'Emotion In' label
        label_out_position = (canvas_size[1] - 120, y0 - dy)  # Position for 'Emotion Out' label

        # Place 'Emotion In' label on the left side
        cv2.putText(canvas_copy, 'Emotion In', label_in_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Place 'Emotion Out' label on the right side
        cv2.putText(canvas_copy, 'Emotion Out', label_out_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Place emotion_in percentages below the 'Emotion In' label
        for i, line in enumerate(emotion_in_percentages):
            y = y0 + i * dy
            cv2.putText(canvas_copy, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Place emotion_out percentages below the 'Emotion Out' label
        for i, line in enumerate(emotion_out_percentages):
            y = y0 + i * dy
            cv2.putText(canvas_copy, line, (canvas_size[1] - 120, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        # Display the canvas with keypoints and connections
        cv2.imshow("Keypoints Visualization", canvas_copy)

        # If saving, write the frame to the video
        if save:
            out.write(canvas_copy)

        # Wait for 100ms and check for "esc" key press to exit
        key = cv2.waitKey(100)
        if key == 27:  
            break

    # Release the video writer, if used
    if save:
        out.release()

    if destroy:
        # Close the visualization window
        cv2.destroyAllWindows()

    
def get_random_frame(data, emotion):
    """
    Get a random frame from a random video of the specified emotion.

    Parameters:
    - data: Nested list representing videos, frames, and features.
    - emotion: Tuple representing the desired emotion in one-hot encoded format.

    Returns:
    - A random frame corresponding to the specified emotion or None if no such frame exists.
    """
    emotion = emotion_to_encoding(emotion)
    # Find indices of videos with the specified emotion
    matching_video_indices = [
        video_idx 
        for video_idx, video in enumerate(data) 
        if tuple(video[0][-7:]) == emotion
    ]
    
    # If no videos match the specified emotion, return None
    if not matching_video_indices:
        return None
    
    # Select a random video
    selected_video_idx = random.choice(matching_video_indices)
    selected_video = data[selected_video_idx]
    
    # Select a random frame
    selected_frame = random.choice(selected_video)
    
    return selected_frame

    
def sanity_check(keypoints):
    """
    Conducts a sanity check on keypoints to ensure biological plausibility.

    Parameters:
    - keypoints (list): A list of (x, y) coordinates for all keypoints.

    Returns:
    - list: List of boolean values indicating the pass status for each keypoint.
    """
    def check_eye_above_nose(eye, nose):
        return eye[1] < nose[1] and eye != (0, 0) and nose != (0, 0)

    def check_ear_above_neck(ear, neck, eye):
        return (ear[1] < neck[1] or (ear[1] >= neck[1] and eye[1] >= neck[1])) and ear != (0, 0) and neck != (0, 0) and eye != (0, 0)
    
    # Define a list of check functions for each keypoint.
    # If no specific check is needed, use None.
    check_functions = [None] * 25
    
    # Assign check for keypoints 15 and 16 (left and right eye) to be above keypoint 0 (nose)
    check_functions[15] = lambda eye: check_eye_above_nose(eye, keypoints[0])
    check_functions[16] = lambda eye: check_eye_above_nose(eye, keypoints[0])
    
    # Assign check for keypoints 17 and 18 (left and right ear) to be above keypoint 1 (neck)
    check_functions[17] = lambda ear: check_ear_above_neck(ear, keypoints[1], keypoints[15])
    check_functions[18] = lambda ear: check_ear_above_neck(ear, keypoints[1], keypoints[16])
    
    # Apply each check function to its corresponding keypoint
    valid_keypoints = [
        check(keypoint) if check is not None else True
        for keypoint, check in zip(keypoints, check_functions)
    ]
    
    return valid_keypoints

def write_notes(notes = None):
    
    if notes is not None:
        # Determine the save path
        save_path = f"D:\\Interactive Dance Thesis Tests\\TransformerResults\\{train_seed}"
        if not os.path.exists(save_path):
            print(f"Creating directory {save_path}")
            os.makedirs(save_path)

        # Determine a unique filename
        existing_files = os.listdir(save_path)
        file_num = 1
        while f"notes_{file_num}.txt" in existing_files:
            file_num += 1
        out_path = os.path.join(save_path, f"notes_{file_num}.txt")

        # Write the notes 
        with open(out_path, 'w') as f:
            f.write(notes)
        print(f"Notes saved to {out_path}")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Training configuration for Motion Model.')
    
    parser.add_argument('--BATCH_SIZE', type=int, default=8, help='Batch size for training.', dest='BATCH_SIZE')
    parser.add_argument('--BLOCK_SIZE', type=int, default=16, help='Block size for sequence processing.', dest='BLOCK_SIZE')
    parser.add_argument('--DROPOUT', type=float, default=0.3, help='Dropout rate for the model.', dest='DROPOUT')
    parser.add_argument('--LEARNING_RATE', type=float, default=0.0001, help='Initial learning rate.', dest='LEARNING_RATE')
    parser.add_argument('--EPOCHS', type=int, default=10000, help='Number of epochs to train.', dest='EPOCHS')
    parser.add_argument('--FRAMES_GENERATE', type=int, default=300, help='Number of frames to generate.', dest='FRAMES_GENERATE')
    parser.add_argument('--TRAIN', action='store_true', help='Flag to indicate training mode.', dest='TRAIN')
    parser.add_argument('--EVAL_EVERY', type=int, default=1000, help='Interval of epochs to perform evaluation.', dest='EVAL_EVERY')
    parser.add_argument('--CHECKPOINT_PATH', type=str, default="checkpoints/proto8_checkpoint.pth", help='Path to save the checkpoint.', dest='CHECKPOINT_PATH')
    parser.add_argument('--L1_LAMBDA', type=float, default=None, help='L1 regularization lambda. Use None for no L1 regularization.', dest='L1_LAMBDA')
    parser.add_argument('--L2_REG', type=float, default=0.0, help='L2 regularization lambda.', dest='L2_REG')
    parser.add_argument('--FINETUNE', action='store_true', help='Flag to indicate fine-tuning mode.', dest='FINETUNE')
    parser.add_argument('--FINE_TUNING_LR', type=float, default=1e-5, help='Learning rate for fine-tuning.', dest='FINE_TUNING_LR')
    parser.add_argument('--FINE_TUNING_EPOCHS', type=int, default=100000, help='Number of epochs for fine-tuning.', dest='FINE_TUNING_EPOCHS')
    parser.add_argument('--PENALTY', action='store_true', help='Flag to indicate if penalty is applied.', dest='PENALTY')
    parser.add_argument('--LATENT_VIS_EVERY', type=int, default=1000, help='Interval of epochs to visualize latent vectors.', dest='LATENT_VIS_EVERY')
    parser.add_argument('--USE_MDN', action='store_true', help='Flag to indicate if MDN layer is used.', dest='USE_MDN')
    parser.add_argument('--notes', type=str, default=None, help='Notes to save with the model.', dest='notes')
    parser.add_argument('--DATASET', type=str, default="MEED", help='Dataset to use.', dest='DATASET')
    parser.add_argument('--PATIENCE', type=int, default=10000, help='Patience for early stopping.', dest='PATIENCE')
    
    args = parser.parse_args()
    
    if args is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(args)
        
    return args

# Define a function to set global variables
def set_globals(args):
    global PATIENCE, BATCH_SIZE,DATASET, BLOCK_SIZE, DROPOUT, LEARNING_RATE, EPOCHS, FRAMES_GENERATE, TRAIN, EVAL_EVERY, CHECKPOINT_PATH, L1_LAMBDA, L2_REG, FINETUNE, FINE_TUNING_LR, FINE_TUNING_EPOCHS, PENALTY, LATENT_VIS_EVERY, notes,USE_MDN
    BATCH_SIZE = args.BATCH_SIZE
    BLOCK_SIZE = args.BLOCK_SIZE
    DROPOUT = args.DROPOUT
    LEARNING_RATE = args.LEARNING_RATE
    EPOCHS = args.EPOCHS
    FRAMES_GENERATE = args.FRAMES_GENERATE
    TRAIN = args.TRAIN
    EVAL_EVERY = args.EVAL_EVERY
    CHECKPOINT_PATH = args.CHECKPOINT_PATH
    L1_LAMBDA = args.L1_LAMBDA
    L2_REG = args.L2_REG
    FINETUNE = args.FINETUNE
    FINE_TUNING_LR = args.FINE_TUNING_LR
    FINE_TUNING_EPOCHS = args.FINE_TUNING_EPOCHS
    PENALTY = args.PENALTY
    LATENT_VIS_EVERY = args.LATENT_VIS_EVERY
    USE_MDN = args.USE_MDN
    DATASET = args.DATASET
    notes = args.notes
    PATIENCE = args.PATIENCE
    
    # ---------------------------------
    notes = f"""Proto8 - trying to adapt Pette et al 2019, addign latent visualisation and analysing latent space. Might be slow, maybe take this out when live.

    
    Added MDN layer to model.
    
    All data, added 10% noise to emotions so model is less stuck. With LeakyRelu
    Loss = mse_loss(keypoints) + mse_loss(emotions) because before output emotions ( which feature was added to keypoint features) were not being matched to input emotions
    No penalty.

    Added dropout to keypoints, also changed input to emotion linear to x and not just emotion (emotion + keypoints)
    Taking extra dropout for emotions and keypoints out, because want model to rely on both equally so what's the point

    dropout keypoints and dropout emotion is currently equal but might change this.

    Emotions and keypoints are multimodal and added separately, but features are added in block processing using +.


    Got rid of both L1 and L2, increasing dropout because model acting weird, this is now delta + coord. 
    Delta is between next frame and current frame. So current frame is previous coord+previous delta. Last frame's delta is 0. 
    
    {BATCH_SIZE} batch size, {BLOCK_SIZE} block size, {DROPOUT} dropout, {LEARNING_RATE} learning rate, {EPOCHS} epochs, {FRAMES_GENERATE} frames generated, {TRAIN} train, {EVAL_EVERY} eval every, {CHECKPOINT_PATH} checkpoint path, {L1_LAMBDA} L1 lambda, {L2_REG} L2 reg"""
    # ---------------------------------
    
    # Print the values using f-string for formatting
    print(f"""
    Batch size set to: {BATCH_SIZE}
    Block size set to: {BLOCK_SIZE}
    Dropout rate set to: {DROPOUT}
    Learning rate set to: {LEARNING_RATE}
    Number of epochs set to: {EPOCHS}
    Frames to generate set to: {FRAMES_GENERATE}
    Training mode set to: {TRAIN}
    Evaluation every set to: {EVAL_EVERY}
    Checkpoint path set to: {CHECKPOINT_PATH}
    L1 regularization lambda set to: {L1_LAMBDA}
    L2 regularization lambda set to: {L2_REG}
    Fine-tuning mode set to: {FINETUNE}
    Fine-tuning learning rate set to: {FINE_TUNING_LR}
    Fine-tuning epochs set to: {FINE_TUNING_EPOCHS}
    Penalty flag set to: {PENALTY}
    Latent visualization every set to: {LATENT_VIS_EVERY}
    Use MDN flag set to: {USE_MDN}
    Dataset set to: {DATASET}
    Patience: {PATIENCE}
    """)
    
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def main(args = None):
    
    # If args are provided, use those; otherwise, parse from command line
    if args is None:
        args = parse_args()

    # Set the global variables based on args
    set_globals(args)

    # Launch tensor board
    writer = SummaryWriter('runs/motion_model_experiment')
    
    processed_data= prep_data(dataset=DATASET)
    global train_data,train_emotions, val_data, val_emotions, frame_dim, max_x, min_x, max_y, min_y, max_dx, min_dx, max_dy, min_dy, threshold
    train_data, train_emotions, val_data, val_emotions, frame_dim, max_x, min_x, max_y, min_y, max_dx, min_dx, max_dy, min_dy, threshold = processed_data
    
    best_val_loss = float('inf')  # Initialize with infinity, so first instance is saved
    epochs_since_improvement = 0  # Counter for epochs since last improvement
    
    # create model
    global m
    m = MotionModel(input_dim=frame_dim, output_dim=frame_dim,emotion_dim=7, blocksize=BLOCK_SIZE, hidden_dim=512, n_layers=8, dropout=DROPOUT)
    m = m.to(device)
    # summary(m, input_size=(BATCH_SIZE, BLOCK_SIZE, frame_dim))
    
    
    global train_seed
    
    if FINETUNE:
        
        # Set lower learning rate for fine-tuning
        optimizer = torch.optim.Adam(m.parameters(), lr=FINE_TUNING_LR, weight_decay=L2_REG)
        global EPOCHS
        EPOCHS = FINE_TUNING_EPOCHS
        
        # Load pre-trained model weights for fine-tuning
        m, optimizer,scheduler, epoch, loss, train_seed = load_checkpoint(m, optimizer, CHECKPOINT_PATH)
        
        print('Fine-tuning model...')
    else:
        # Training from scratch
        optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
        

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    #loss evaluated at every EVAL EVERY, so 2*EVAL_EVERY is scheduler's actual patience
    # train
    if TRAIN or FINETUNE:
        # Generate a random seed
        
        train_seed = random.randint(1, 100000)
        print(f'Training model {train_seed}...')
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')  # Initialize with infinity, so first instance is saved
    
        # List to store latent vectors
        latent_space = [] 
        
        for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch"):
            # get sample batch of data
            xb, yb, eb, _ = get_batch("train", BLOCK_SIZE, BATCH_SIZE, train_data,train_emotions, val_data, val_emotions)
          
            # evaluate loss
            if USE_MDN:
                pi, sigma, mu, logits, emotion_logits, loss, latent_vectors = m(xb, yb, eb)
                mse_logits_loss, mse_emotion_loss, mdn_loss = loss
                total_loss = mse_logits_loss + mse_emotion_loss + mdn_loss
            else:
                logits,emotion_logits,loss,latent_vectors = m(xb,yb,eb)
                mse_logits_loss, mse_emotion_loss = loss
                total_loss = mse_logits_loss + mse_emotion_loss
                mdn_loss = 0
            
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            
            # Log the losses
            writer.add_scalar('Loss/Total', mse_logits_loss.item(), epoch)
            writer.add_scalar('Loss/Emotion', mse_emotion_loss.item(), epoch)
            if USE_MDN:
                writer.add_scalar('Loss/MDN', mdn_loss.item(), epoch)
            
            # Clip gradients
            # nn.utils.clip_grad_norm_(m.parameters(), max_norm=1)

            
            optimizer.step()
            
            # evaluate and save loss
            if epoch % EVAL_EVERY == 0:
                losses = estimate_loss()
                scheduler.step(losses['val'])
                print(f"\nTrain loss: {losses['train']:.6f} val loss: {losses['val']:.6f}")
                global notes
                notes += f"""\nEPOCH:{epoch} \nTrain loss: {losses['train']:.6f} val loss: {losses['val']:.6f}"""
                if (epoch != 0):
                    # Store the losses for plotting
                    train_losses.append(losses['train'])
                    val_losses.append(losses['val'])
                
                if (epoch % (EVAL_EVERY*10) == 0) and (epoch != 0):
                    # Save a checkpoint every 10 rounds of eval if it has the best validation loss so far
                    if losses['val'] < best_val_loss:
                        print(f"-> Best model so far (val loss: {best_val_loss:.6f}), saving model...")
                        best_val_loss = losses['val']
                        save_checkpoint(model=m, optimizer=optimizer, scheduler=scheduler,epoch=epoch, loss=loss, checkpoint_path=CHECKPOINT_PATH)
                        print(f'Model {train_seed} saved!')
                        epochs_since_improvement = 0  
                    else:
                        # Multiple of EVAL_EVERY
                        epochs_since_improvement += 1

                    if (epochs_since_improvement >= PATIENCE) and (PATIENCE != 0):
                        print(f"No improvement in validation loss for {PATIENCE} consecutive epochs. Stopping training.")
                        break

                # If you want to collect and visualize latent vectors periodically during training:
                if epoch % LATENT_VIS_EVERY == 0:
                    with torch.no_grad():
                        m.eval()  # Switch to evaluation mode
                        # Get a batch of data to visualize latent space
                        xb, yb, eb, _ = get_batch("val", BLOCK_SIZE, BATCH_SIZE, train_data, train_emotions, val_data, val_emotions)
                        
                        if USE_MDN:
                            _,_,_, _, _, _, latent_vectors = m(xb, yb, eb)
                        
                        else: 
                            _,_,_,latent_vectors = m(xb,yb,eb)
                            
                        latent_array = latent_vectors.cpu().detach().clone().numpy()
                        latent_space.append(latent_array)
                        m.train()  # Switch back to training mode
            
            
        # After the training loop, save the final model
        try:
            if val_losses[-1] < best_val_loss:
                print(f"-> Best model so far (val loss: {best_val_loss:.6f}), saving model...")
                best_val_loss = val_losses[-1]
                save_checkpoint(model=m, optimizer=optimizer, scheduler=scheduler, epoch=EPOCHS, loss=val_losses[-1], checkpoint_path=CHECKPOINT_PATH)
                print(f'Model {train_seed} saved!')
        
        except IndexError:
            print('No validation losses to save!')
        # After the training loop, plot the losses
        plot_losses(train_losses, val_losses, EPOCHS, EVAL_EVERY,train_seed)
        
        # Concatenate all collected latent vectors
        latent_space = np.concatenate(latent_space, axis=0)
        
    else:
        # Load the model
        print('Loading model...')
        m, optimizer, scheduler, epoch, loss, train_seed = load_checkpoint(m, optimizer, CHECKPOINT_PATH,scheduler)
        print(f"Model {train_seed} loaded from {CHECKPOINT_PATH} (epoch {epoch}, loss {loss:.6f})")
    
    

    # Generate a sequence
    print(f'Generating sequence of {FRAMES_GENERATE} frames...')
    # xb and yb should always have the same emotion - because same video
    xb, yb, eb, _ = get_batch("val", BLOCK_SIZE, BATCH_SIZE, train_data,train_emotions, val_data, val_emotions)

    emotion_in = torch.as_tensor(add_noise_to_emotions(eb.cpu().numpy(), noise_level=0.2), dtype=torch.float32).to(device)
    B, T, C = xb.shape
    random_indices = torch.randint(0, T, (B,))
    xb_random_frame = xb[torch.arange(B), random_indices].unsqueeze(1) 
    # generated_keypoints,generated_emotion = m.generate(xb_random_frame, emotion_in, FRAMES_GENERATE)
    generated_keypoints,generated_emotion = m.generate(xb, emotion_in, FRAMES_GENERATE)
    # unnorm_out = unnormalise_list_2D(generated, max_x, min_x, max_y, min_y,max_dx, min_dx, max_dy, min_dy)
    
    
    # # ====
    # print(generated_keypoints)
    
    # output_file_path = 'output_file.txt'  # Replace with your desired file path

    # generated_keypoints, generated_emotion = m.generate(xb, emotion_in, FRAMES_GENERATE)
    # # print(generated_keypoints)

    # # Check if the file exists. If not, create it.
    # if not os.path.exists(output_file_path):
    #     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # # Write the output to the file
    # with open(output_file_path, 'w') as file:
    #     file.write(str(generated_keypoints))
    #     file.write('\n')
    #     file.write(str(generated_emotion))
        
    # # ====
        
    unnorm_out = unnormalise_list_2D(generated_keypoints, max_x, min_x, max_y, min_y,max_x, min_x, max_y, min_y)
    # unnorm_out = unnormalise_list_2D(xb, max_x, min_x, max_y, min_y,max_x, min_x, max_y, min_y)
    
    # visualise and save
    for i,batch in enumerate(unnorm_out):
        
        emotion_vectors = (emotion_in[i],generated_emotion[i])
        frame = int(random.randrange(len(batch))) #only feed 1 frame in for visualisation
        visualise_skeleton(batch, max_x, max_y,emotion_vectors, max_frames=FRAMES_GENERATE,save = True,save_path=None,prefix=f'{EPOCHS}_mdnsample_sigma100',train_seed=train_seed,delta=False)
        # visualise_skeleton(batch, max_x, max_y,emotion_vectors, max_frames=FRAMES_GENERATE,save = True,save_path=None,prefix=f'adam_{EPOCHS}_delta',train_seed=train_seed,delta=True)


    if TRAIN or FINETUNE:
        write_notes(notes)
     
    print('Done!')
    
    if TRAIN or FINETUNE:
        return latent_space, train_seed
    else:
        return None, train_seed
    
if __name__ == "__main__":
    # Define the arguments you want to pass
    args = argparse.Namespace(
        BATCH_SIZE=8,
        BLOCK_SIZE=16,
        DROPOUT=0.2,
        LEARNING_RATE=0.0001,
        EPOCHS=3000,
        FRAMES_GENERATE=300,
        TRAIN=True,
        EVAL_EVERY=1000,
        CHECKPOINT_PATH="checkpoints/proto8_checkpoint_temp.pth",
        L1_LAMBDA=None,
        L2_REG=0.0,
        FINETUNE=False,
        FINE_TUNING_LR=1e-5,
        FINE_TUNING_EPOCHS=100000,
        PENALTY=False,
        LATENT_VIS_EVERY=1000,
        USE_MDN = True,
        PATIENCE= 5, #multiple of EVAL_EVERY * 10 
        DATASET = "all",
        
        # NOTES---------------------------------
        notes = f"""Proto8 - trying to adapt Pette et al 2019, addign latent visualisation and analysing latent space. Might be slow, maybe take this out when live.
        
        Added MDN to increase variance of output as Bishop et al 1994. and Alemi et al 2017.
        
        Updated loss to loss = F.mse_loss(logits, targets) + (F.mse_loss(emotion_logits, emotions)) + mdn.mdn_loss(pi, sigma, mu, targets)
        see if that will help with noise
        
        
        convert from random sampling MDN to find the index of the most probable Gaussian component hopefully will lead to smoother outputs
        
        adjusted sampling to /100 of sigma, hopefully will lead to smoother outputs
        
        all data

        All data, added 10% noise to emotions so model is less stuck. With LeakyRelu
        Loss = mse_loss(keypoints) + mse_loss(emotions) because before output emotions ( which feature was added to keypoint features) were not being matched to input emotions
        No penalty.

        Added dropout to keypoints, also changed input to emotion linear to x and not just emotion (emotion + keypoints)
        Taking extra dropout for emotions and keypoints out, because want model to rely on both equally so what's the point

        dropout keypoints and dropout emotion is currently equal but might change this.

        Emotions and keypoints are multimodal and added separately, but features are added in block processing using +.


        Got rid of both L1 and L2, increasing dropout because model acting weird, this is now delta + coord. 
        Delta is between next frame and current frame. So current frame is previous coord+previous delta. Last frame's delta is 0. 
        """
    )
    latent_space, train_seed = main(args)