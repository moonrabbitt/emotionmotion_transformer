# Changed model to delta model
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger()
# Clear previous handlers
for handler in logger.handlers[:]:
    handler.close()
    logger.removeHandler(handler)

logging.basicConfig(filename= f"preprocessing_log.txt", level=logging.INFO, filemode='w')
# logging clear file
logger = logging.getLogger()

# Set root directory
root_dir = "C:\\Users\\avika\\OneDrive\\Documents\\UAL\\interactive_dance_thesis"
os.chdir(root_dir)

# Check if the current working directory was set correctly
print(os.getcwd())

# Functions--------------------------------------------------
# Load and preprocess data------------------------------------

def interpolate(coord_prev, coord_next):
    """
    Linearly interpolate between two coordinates.
    
    Parameters:
    - coord_prev (float): Coordinate of the previous frame.
    - coord_next (float): Coordinate of the next frame.  

    Returns:
    - (float): Interpolated coordinate.
    """
    return (coord_prev + coord_next) / 2

def preprocess_data(files: List[str]) -> dict:
    """
    Pre-process data by interpolating to avoid (0,0) keypoints.

    Parameters:
    - files (List[str]): List of file paths to process.

    Returns:
    - dict: Pre-processed data.
    """
    x_list=[]
    y_list=[]
    conf_list=[]
    emotions = []
    
    for file in tqdm(files):
        with open(file) as f:
            data = json.load(f)
            x = data['x']
            y = data['y']
            conf = data['confidence']
            emotion_code = [file.split('_')[-2].split('\\')[0][3:-3]]
            emotions.extend(emotion_code)
            
            if len(emotion_code) > 1:
                print(emotion_code)
            
            for i in range(len(x)):
                # Check if coordinate is (0,0)
                if x[i] == 0 and y[i] == 0:
                    # logger.info(f"Found (0,0) at index {i} in file {file}")
                    # If first frame, copy from next frame
                    if i == 0:
                        j = i + 1
                        # Find next non-(0,0) frame
                        while x[j] == 0 and y[j] == 0:
                            j += 1
                        x[i] = x[j]
                        y[i] = y[j]
                    # If last frame, copy from previous frame
                    elif i == len(x) - 1:
                        x[i] = x[i-1]
                        y[i] = y[i-1]
                    # For a frame in the middle
                    else:
                        # Find the next non-(0,0) frame
                        j = i + 1
                        while j < len(x) and x[j] == 0 and y[j] == 0:
                            j += 1
                        # If no non-(0,0) frame found, use the previous frame, otherwise interpolate
                        if j == len(x):
                            x[i] = x[i-1]
                            y[i] = y[i-1]
                        else:
                            x[i] = interpolate(x[i-1], x[j])
                            y[i] = interpolate(y[i-1], y[j])
            
            x_list.append(x)
            y_list.append(y)
            conf_list.append(conf)

    return {"x": x_list, "y": y_list, "confidence": conf_list, "emotions": emotions}

def validate_interpolation(x_list,y_list,files):
    print("Validating interpolation...")
    err = 0
    for i in range(len(x_list)):
        for j in range(len(x_list[i])):
            if x_list[i][j] == 0 and y_list[i][j] == 0:
                print(f"Found (0,0) at index {j} in file {files[i]}")
                err += 1
        
    if err == 0:
        print("No errors found!")
        
    return err == 0
                


# Prepare data for training------------------------------------
def normalize_values_2D(frames):
    """
    Takes in a list of lists (frames), returns max and min values and normalized list
    
    Parameters:
        frames: List of lists containing keypoints for each frame.
    
    Returns:
        max_val: Maximum keypoint value across all frames.
        min_val: Minimum keypoint value across all frames.
        normalized_frames: Normalized keypoints for each frame.
    """
    # Flatten the data to find global min and max
    flat_data = [kp for frame in frames for kp in frame]
    max_val = max(flat_data)
    min_val = min(flat_data)
    
    # Normalize data
    normalized_frames = [
        [2 * (kp - min_val) / (max_val - min_val) - 1 for kp in frame]
        for frame in frames
    ]
    
    return max_val, min_val, normalized_frames

def create_kp_frames(normalised_x, normalised_y):
    """Create 1D array of 50 numbers (x,y,x,y --> 25 keypoints) from 2D array for each frame for transformer input, returned as list of lists"""
    
    print("Creating keypoint frames...")
    kp_frames = []
    n_parts = 25

    for i in tqdm(range(0, len(normalised_x))):
        video_x = normalised_x[i]
        video_y = normalised_y[i]
        kp_frame= []
        for j in range(0,len(video_x), n_parts):
            frame_data = [coord for pair in zip(video_x[j:j+n_parts], video_y[j:j+n_parts]) for coord in pair]
            kp_frame.append(frame_data)
        kp_frames.append(kp_frame)


    return kp_frames

def stratified_split(data, test_size=0.1):
    # Organize data by class
    class_data = {}
    for video_index, video in enumerate(data):
        # Assume the last 7 elements of the first frame of each video represent the class (emotion)
        emotion = tuple(video[0][-7:])  
        if emotion not in class_data:
            class_data[emotion] = []
        class_data[emotion].append(video_index)  # Store video index instead of data to save memory

    train_indices = []
    val_indices = []

    # For each class, split the data into train and val sets
    for emotion, video_indices in class_data.items():
        random.shuffle(video_indices)  # Shuffle indices to ensure random splits
        split_idx = int(len(video_indices) * (1 - test_size))  # Index to split train and val
        train_indices.extend(video_indices[:split_idx])
        val_indices.extend(video_indices[split_idx:])

    # Retrieve the data using the indices
    train_data = [data[idx] for idx in train_indices]
    val_data = [data[idx] for idx in val_indices]

    # Shuffle the train and val sets to ensure random order
    random.shuffle(train_data)
    random.shuffle(val_data)

    return train_data, val_data

def get_batch(split, block_size, batch_size, device=device):
    data = train_data if split == 'train' else val_data
    
    # Choose random videos
    ix = torch.randint(len(data), (batch_size,))

    # For each chosen video, select a random starting point
    start_frames = [torch.randint(len(data[i]) - block_size, (1,)).item() for i in ix]

    # Extract subsequences from each chosen video and convert to tensors
    x = torch.stack([torch.tensor(data[i][start:start+block_size], dtype=torch.float32) for i, start in zip(ix, start_frames)])
    y = torch.stack([torch.tensor(data[i][start+1:start+block_size+1], dtype=torch.float32) for i, start in zip(ix, start_frames)])

    # Compute the mask to mask out -inf values
    mask = (x != float('-inf')).all(dim=-1).float()  # assuming -inf is present in any part of the data point

    # Move tensors to the designated device
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    
    return x, y, mask

# Dealing with emotions------------------------------------
def emotion_labels_to_vectors(emotion_labels):
    """
    Convert a list of emotion labels to a list of continuous emotion vectors.

    Parameters:
    - emotion_labels (list of str): A list of emotion labels.
    
    Returns:
    - list of np.array: A list of continuous emotion vectors.
    """
    # Define a mapping from emotion labels to emotion vectors
    label_to_vector = {
        'A': [1, 0, 0, 0, 0, 0, 0],   # Anger
        'D': [0, 1, 0, 0, 0, 0, 0],   # Disgust
        'F': [0, 0, 1, 0, 0, 0, 0],   # Fear
        'H': [0, 0, 0, 1, 0, 0, 0],   # Happiness
        'N': [0, 0, 0, 0, 1, 0, 0],   # Neutral
        'SA': [0, 0, 0, 0, 0, 1, 0],  # Sad
        'SU': [0, 0, 0, 0, 0, 0, 1]   # Surprise
    }
    
    # Convert the labels to vectors using the mapping
    emotion_vectors = [label_to_vector[label] for label in emotion_labels]
    
    return emotion_vectors

def add_emotions_to_frames(kp_frames, emotion_vectors):
    # kp_frames is normalised
    print("Adding emotions to frames...")
    kp_frames_with_emotion = []
    
    for i in tqdm(range(len(emotion_vectors))):
    # Use list concatenation instead of extend() to avoid in-place modification and None
        for frame in kp_frames[i]:
            frame.extend(emotion_vectors[i])
        kp_frames_with_emotion.append(kp_frames[i])
    
    if len(kp_frames_with_emotion) != len(kp_frames):
        raise Exception("Error: number of frames with emotion does not match number of frames without emotion")
        
    return kp_frames_with_emotion

def validate_emotion_consistency(x, y):
    """
    Validate that the emotion code (last 7 elements of each frame) is consistent
    between corresponding frames in x and y.

    Parameters:
    - x (Tensor): Input sequences (batch_size, sequence_length, frame_length)
    - y (Tensor): Target sequences (batch_size, sequence_length, frame_length)
    
    Returns:
    - bool: True if emotions are consistent, False otherwise
    """
    # Extract the emotion encodings from x and y
    emotion_x = x[:, :, -7:]
    emotion_y = y[:, :, -7:]

    # Check if the emotion encodings are equal in x and y
    emotion_equal = torch.all(emotion_x == emotion_y, dim=-1)
    
    # Check equality across sequence length (assuming dim 1 is sequence_length)
    emotion_equal = torch.all(emotion_equal, dim=-1)

    # Check if all batches have consistent emotions
    all_equal = torch.all(emotion_equal)

    is_consistent = all_equal.item()
        
    if not is_consistent:
        raise Exception("Emotions are inconsistent between x and y.")



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
    
    def __init__(self,head_size,n_emb,dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False, device=device)
        self.query = nn.Linear(n_emb, head_size, bias=False, device=device)
        self.value = nn.Linear(n_emb, head_size, bias=False, device=device)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.n_emb = n_emb
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape # batch size, time, context
        # key, query, value
        k = self.key(x) # B,T,C
        q = self.query(x) # B,T,C
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
        
    def forward(self, x):
        # x is (B,T,C)
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,C*num_heads)
        out = self.dropout(self.proj(out)) # (B,T,C) - projection back into residual pathway
        
        return out
    
class FeedForward(nn.Module):
    """A simple lineear layer followed by a ReLU - allows all tokens to think on data individually"""
    
    def __init__(self,n_emb,dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb , device=device), # 4 * because recommended in paper residual pathway - growing residual pathway
            nn.ReLU(),
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
        self.ln1 =  nn.LayerNorm(n_emb , device=device)
        self.ln2 =  nn.LayerNorm(n_emb, device=device)
        
    def forward(self, x):
        # x + due to residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class MotionModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_layers=8 , dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False, device=device) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False,device=device)
        # PROBLEM
        self.positional_encoding = positional_encoding(seq_len=BLOCK_SIZE, d_model=hidden_dim).to(device)
        layers = [Block(n_emb=hidden_dim, n_heads=4) for _ in range(n_layers)]
        layers.append(nn.LayerNorm(hidden_dim, device=device))
        # layers.append(nn.InstanceNorm1d(hidden_dim, device=device))
        self.blocks = nn.Sequential(*layers)

        self.lm_head = nn.Linear(hidden_dim, hidden_dim, bias=False, device=device)
       
    
        
    def forward(self, inputs, targets=None , l1_lambda = 0.001, mask=None,):
        B,T,C = inputs.shape # batch size, time, context
        
        # fc1 transforms input into hidden dimension
        x = self.fc1(inputs) # B,T,hidden dimension
        # Add positional encoding
       
        x += positional_encoding(seq_len=T, d_model=self.hidden_dim).to(device) # positional_encoding = T,hidden dimension , added = B,T,hidden dimension
        
        x = self.blocks(x) # B,T,hidden dimension
        x = self.lm_head(x) # B,T,hidden dimension
        
        # fc2 transforms hidden dimension into output dimension
        logits = self.fc2(x)
        
        
        if targets is None:
            loss = None
        
        else:
            B,T,C = inputs.shape # batch size, time, context
            # You can adjust this value based on your needs
           
            if L1_LAMBDA is None:
                loss = F.mse_loss(logits, targets) # mse picked cause continous data
                
            else:
                l1_norm = sum(p.abs().sum() for p in m.parameters())  # Calculate L1 norm for all model parameters
                loss = F.mse_loss(logits, targets) + l1_lambda * l1_norm

            # adding mask to ignore 0,0 occlusions (-inf)
            # if mask is None:
            #     mask = (inputs != float('-inf')).all(dim=-1).float() 
              
            # loss = F.mse_loss(logits * mask.unsqueeze(-1), targets * mask.unsqueeze(-1), reduction='sum') / mask.sum()

        
        return logits,loss
    
    def generate(self,inputs,max_new_tokens):
        # inputs is (B,T) array of indices in current context
        # get current prediction
    
        generated_sequence = inputs
        
        for _ in range(max_new_tokens):
            cond_sequence = generated_sequence[:, -BLOCK_SIZE:] # get the last block_size tokens from the generated sequence so positional doesn't run out
            # don't actually need to do this cause positional is sinusoidal but just in case since model trained with blocksize
            logits, _ = self(cond_sequence)
            next_values = logits[:, -1, :]  # Get the values from the last timestep
            
            # Append the predicted values to the sequence
            generated_sequence = torch.cat([generated_sequence, next_values.unsqueeze(1)], dim=1)
        
        return generated_sequence
    
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
            xb, yb, _ = get_batch(split, BLOCK_SIZE, BATCH_SIZE)
            _, loss = m(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

# test----------------------------------------------------
def unnormalise_list_2D(data_tensor, max_x, min_x, max_y, min_y):
    all_frames = []
    # Loop through each batch
    for batch_idx in range(data_tensor.size(0)):
        batch_frames = []
        # Loop through each frame in the batch
        for frame_idx in range(data_tensor.size(1)):
            frame_data = data_tensor[batch_idx, frame_idx, :]
            unnormalized_data = []
            # Loop through the coordinate pairs and unnormalize
            for i in range(0, 50, 2):  
                x = frame_data[i]
                y = frame_data[i+1]
                unnormalized_x = (x+1)/2 * (max_x-min_x) + min_x
                unnormalized_y = (y+1)/2 * (max_y-min_y) + min_y
                unnormalized_data.extend([unnormalized_x.item(), unnormalized_y.item()])
            # Append the emotion encoding without unnormalizing
            unnormalized_data.extend(frame_data[-7:].tolist())
            batch_frames.append(unnormalized_data)
        all_frames.append(batch_frames)
    return all_frames


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Steps (in thousands)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Use the run seed in the filename
    plot_path = os.path.join("D:/Interactive Dance Thesis Tests/TransformerResults/losses", f"loss_plot_{train_seed}.png")
    
    # Save the plot
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Save the model checkpoint."""
    # Use the run seed in the filename
    # checkpoint_path = os.path.join(checkpoint_dir, f"MEED_checkpoint_{run_seed}.pth")
    
    
    if not os.path.exists(checkpoint_path):
        print('Creating checkpoints directory...')
        os.makedirs(checkpoint_path)
    
    print(f"Saving model checkpoint to {checkpoint_path}")
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch,
             'loss': loss,
             'train_seed' : train_seed}
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load the model checkpoint."""
    print('Loading checkpoint...')
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    epoch = state['epoch']
    loss = state['loss']
    train_seed = state['train_seed']
    print(f"Checkpoint loaded from {checkpoint_path}")
    return model, optimizer, epoch, loss,train_seed
    
def visualise_skeleton(all_frames, max_x, max_y, max_frames=500, save=False, save_path=None, prefix=None):
    
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
    


    # Define the codec and create VideoWriter object
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

    # Iterate over every frame
    for frame_data in all_frames[:max_frames]:
        
        
        canvas_copy = canvas.copy()

        # Extract x and y coordinates
        x_coords = frame_data[0:50:2] 
        y_coords = frame_data[1:50:2]
        emotion_vector = tuple(frame_data[-7:])
        

        # Get emotion percentages and labels
        emotion_percentages = [f"{int(e * 100)}% {label}" for e, label in zip(emotion_vector, emotion_labels) if e > 0]

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
        y0, dy = 30, 15  # Starting y position and line gap
        for i, line in enumerate(emotion_percentages):
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

    cv2.destroyAllWindows()

    
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

if __name__ == "__main__":
    
    # load and preprocess data
    direction = ['left','right','front']
    files = []

    for d in direction:
        files.extend(glob.glob(f"G:/UAL_Thesis/affective_computing_datasets/multiview-emotional-expressions-dataset/*/{d}_*/processed_data.json"))

    processed_data = preprocess_data(files)
    
    x_list = processed_data['x']
    y_list = processed_data['y']
    conf_list = processed_data['confidence']
    emotions_labels = processed_data['emotions']

    if validate_interpolation(processed_data['x'],processed_data['y'],files) == False:
        raise Exception('Interpolation not successful')
    
    # prepare data for training
    global max_x, min_x, max_y, min_y
    
    max_x, min_x, normalised_x = normalize_values_2D(x_list)
    max_y, min_y, normalised_y = normalize_values_2D(y_list)
   
    kp_frames = create_kp_frames(normalised_x, normalised_y)  # 1D tensor array of 50 numbers (x,y,x,y --> 25 keypoints)
    data = add_emotions_to_frames(kp_frames, emotion_labels_to_vectors(emotions_labels))
    train_data, val_data = stratified_split(data, test_size=0.1)
  
    
    # HYPERPARAMETERS------------------
    
    torch.manual_seed(1337)
    BATCH_SIZE = 4 # how many independent sequences will we process in parallel? - every forward and backward pass in transformer
    BLOCK_SIZE = 16 # what is the maximum context length for predictions? 
    DROPOUT = 0.3
    LEARNING_RATE = 0.0001
    EPOCHS = 300000
    FRAMES_GENERATE = 300
    TRAIN = True
    EVAL_EVERY = 1000
    CHECKPOINT_PATH = "checkpoints/proto4_checkpoint.pth"
    L1_LAMBDA = None
    L2_REG = 0.0
    # L2_REG=0.0
    global train_seed
    
    # ---------------------------------
    
    # NOTES---------------------------------
    notes = f"""Got rid of both L1 and L2, increasing dropout because model acting weird"""
    # ---------------------------------
    
    # create model
    frame_dim = 57 # how many numbers are in each frame? - 50 kps xy + 7 emotion encodings
    m = MotionModel(input_dim=frame_dim, output_dim=frame_dim, hidden_dim=512, n_layers=8, dropout=DROPOUT)
    m = m.to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)  # weight_decay=1e-5 L2 regularization
    
    # train
    if TRAIN:
        # Generate a random seed
        
        train_seed = random.randint(1, 100000)
        print(f'Training model {train_seed}...')
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')  # Initialize with infinity, so first instance is saved
    
        
        for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch"):
            # get sample batch of data
            xb,yb,mask = get_batch('train',BLOCK_SIZE,BATCH_SIZE)
            # validate emotions are consistent
            validate_emotion_consistency(xb, yb)
            # evaluate loss
            logits, loss = m(xb,yb,l1_lambda=L1_LAMBDA)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if epoch % EVAL_EVERY == 0:
                losses = estimate_loss()
                print(f"\nTrain loss: {losses['train']:.6f} val loss: {losses['val']:.6f}")
                
                if (epoch != 0):
                    # Store the losses for plotting
                    train_losses.append(losses['train'])
                    val_losses.append(losses['val'])
                
                if (epoch % (EVAL_EVERY*10) == 0) and (epoch != 0):
                    # Save a checkpoint every 10 rounds of eval if it has the best validation loss so far
                    if losses['val'] < best_val_loss:
                        print(f"-> Best model so far (val loss: {best_val_loss:.6f}), saving model...")
                        best_val_loss = losses['val']
                        save_checkpoint(model=m, optimizer=optimizer, epoch=epoch, loss=loss, checkpoint_path=CHECKPOINT_PATH)
                        print(f'Model {train_seed} saved!')
                        
        # After the training loop, save the final model
        if val_losses[-1] < best_val_loss:
                        print(f"-> Best model so far (val loss: {best_val_loss:.6f}), saving model...")
                        best_val_loss = val_losses[-1]
                        save_checkpoint(model=m, optimizer=optimizer, epoch=EPOCHS, loss=val_losses[-1], checkpoint_path=CHECKPOINT_PATH)
                        print(f'Model {train_seed} saved!')
        # After the training loop, plot the losses
        plot_losses(train_losses, val_losses)
        
    else:
        # Load the model
        print('Loading model...')
        m, optimizer, epoch, loss, train_seed = load_checkpoint(m, optimizer, CHECKPOINT_PATH)
        print(f"Model {train_seed} loaded from {CHECKPOINT_PATH} (epoch {epoch}, loss {loss:.6f})")
    
    # Generate a sequence
    print(f'Generating sequence of {FRAMES_GENERATE} frames...')
    xb,yb,mask = get_batch('val',BLOCK_SIZE,BATCH_SIZE)

    generated = m.generate(xb, FRAMES_GENERATE)
    unnorm_out = unnormalise_list_2D(generated, max_x, min_x, max_y, min_y)
    
    # visualise and save
    for batch in unnorm_out:
        visualise_skeleton(batch, max_x, max_y, max_frames=FRAMES_GENERATE,save = True,save_path=None,prefix=f'adam_{EPOCHS}')
    
    if TRAIN:
        write_notes(notes)
     
    print('Done!')