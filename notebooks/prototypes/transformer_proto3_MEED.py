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

# Functions
# Load and preprocess data

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
    print("Preprocessing data...")
    for file in tqdm(files):
        with open(file) as f:
            data = json.load(f)
            x = data['x']
            y = data['y']
            conf = data['confidence']
            
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
            
            x_list.extend(x)
            y_list.extend(y)
            conf_list.extend(conf)

    return {"x": x_list, "y": y_list, "confidence": conf_list}


# Prepare data for training
def normalise_values(_list):
    """Takes in list, returns max and min values and normalised list"""
    max_,min_ = max(_list), min(_list)
    normalised = [2 * (_pos - min_) / (max_ - min_) - 1 for _pos in _list]
    
    return max_, min_, normalised

def create_kp_frames(normalised_x, normalised_y):
    """Create 1D array of 50 numbers (x,y,x,y --> 25 keypoints) for each frame for transformer input, returned as tensor"""
    kp_frame = []
    n_parts = 25

    print("Creating keypoint frames...")
    for i in tqdm(range(0, len(normalised_x), n_parts)):
        frame_data = [coord for pair in zip(normalised_x[i:i+n_parts], normalised_y[i:i+n_parts]) for coord in pair]
        kp_frame.append(frame_data)
    print("Returned as tensor...")
    return torch.tensor(kp_frame, dtype= torch.float32)

def get_batch(split,data):
    # Let's now split up the data into train and validation sets
    n = int(0.8*len(data)) # first 80% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    
    data = train_data if split == 'train' else val_data

    # We need to account for the fact that our data is 2-dimensional when creating batches.
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    
    # Adjusted these lines to consider the 50-dimensional poses
    x = torch.stack([data[i:i+BLOCK_SIZE, :] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1, :] for i in ix])
    
    # Compute the mask to mask out -inf values
    mask = (x != float('-inf')).all(dim=-1).float()  # this assumes -inf is present in any part of the data point

    x, y, mask = x.to(device), y.to(device), mask.to(device)
    
    x, y = x.to(device), y.to(device)
    return x, y ,mask

# Decoder Model
# let's start with a very simple model

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

    def __init__(self, n_emb, n_heads):
        
        super().__init__()
        head_size = n_emb//n_heads
        self.sa = MultiHeadAttention(num_heads=n_heads, head_size=head_size, n_emb=n_emb)
        self.ffwd = FeedForward(n_emb=n_emb)
        self.ln1 =  nn.InstanceNorm1d(n_emb , device=device)
        self.ln2 =  nn.InstanceNorm1d(n_emb, device=device)
        
    def forward(self, x):
        # x + due to residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class MotionModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False, device=device) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False,device=device)
        # PROBLEM
        self.positional_encoding = positional_encoding(seq_len=BLOCK_SIZE, d_model=hidden_dim).to(device)
        layers = [Block(n_emb=hidden_dim, n_heads=4) for _ in range(n_layers)]
        layers.append(nn.InstanceNorm1d(hidden_dim, device=device))
        self.blocks = nn.Sequential(*layers)

        self.lm_head = nn.Linear(hidden_dim, hidden_dim, bias=False, device=device)
       
    
        
    def forward(self, inputs, targets=None ,mask=None):
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
            loss = F.mse_loss(logits, targets) # mse picked cause continous data
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
    
# train
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    eval_iters = 100
    print('Evaluating loss...')
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters), desc=f"Evaluating Loss", unit="batch"):
            xb, yb, _ = get_batch(split,data)
            _, loss = m(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

# test
def unnormalise_list(data_list, max_x, min_x, max_y, min_y):
    unnormalized_data = []
    for i in range(0, len(data_list), 2):  # Step by 2 to get x, y pairs
        x = data_list[i]
        y = data_list[i+1]
        
        # Unnormalize x and y
        unnormalized_x = (x+1)/2 * (max_x-min_x) + min_x
        unnormalized_y = (y+1)/2 * (max_y-min_y) + min_y
        
        # Append to result list
        unnormalized_data.extend([unnormalized_x, unnormalized_y])
    
    return unnormalized_data

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
    
def visualise_skeleton(all_frames, max_x, max_y, max_frames=100, save=False, save_path=None, prefix=None):
    
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

    
    # Initialize a blank canvas (image)
    canvas_size = (int(max_y)+50, int(max_x)+50, 3)  
    canvas = np.zeros(canvas_size, dtype=np.uint8)

    # Define the codec and create VideoWriter object
    if save:
        # Determine the save path
        if save_path is None:
            save_path = f"D:\\Interactive Dance Thesis Tests\\TransformerResults\\{train_seed}\\"

        # Ensure directory exists
        if not os.path.exists(save_path):
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
        x_coords = frame_data[0::2] 
        y_coords = frame_data[1::2]

        # Plot keypoints on the canvas
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            x_val = x.item() if torch.is_tensor(x) else x
            y_val = y.item() if torch.is_tensor(y) else y
            cv2.circle(canvas_copy, (int(x_val), int(y_val)), 3, (0, 0, 255), -1)  
            cv2.putText(canvas_copy, keypointsMapping[i], (int(x_val), int(y_val)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw connections (limbs) on the canvas
        for limb in limb_connections:
            start_point = (int(x_coords[keypointsMapping.index(limb[0])]), int(y_coords[keypointsMapping.index(limb[0])]))
            end_point = (int(x_coords[keypointsMapping.index(limb[1])]), int(y_coords[keypointsMapping.index(limb[1])]))

            if start_point == (0,0) or end_point == (0,0):
                continue
            cv2.line(canvas_copy, start_point, end_point, (0, 255, 0), 2)  

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


if __name__ == "__main__":
    
    # load and preprocess data
    files = glob.glob("G:/UAL_Thesis/affective_computing_datasets/multiview-emotional-expressions-dataset/*/front_*/processed_data.json")
    processed_data = preprocess_data(files)
    x_list = processed_data['x']
    y_list = processed_data['y']
    conf_list = processed_data['confidence']

    
    # prepare data for training
    global max_x, min_x, max_y, min_y
    
    max_x, min_x, normalised_x = normalise_values(x_list)
    max_y, min_y, normalised_y = normalise_values(y_list)
   
    data = create_kp_frames(normalised_x, normalised_y)  # 1D tensor array of 50 numbers (x,y,x,y --> 25 keypoints)
    print(data.shape, data.dtype)
    
    # HYPERPARAMETERS------------------
    
    torch.manual_seed(1337)
    BATCH_SIZE = 4 # how many independent sequences will we process in parallel? - every forward and backward pass in transformer
    BLOCK_SIZE = 16 # what is the maximum context length for predictions? 
    LEARNING_RATE = 0.0001
    EPOCHS = 1000
    FRAMES_GENERATE = 100
    TRAIN = True
    EVAL_EVERY = 100
    CHECKPOINT_PATH = "checkpoints/MEED_checkpoint.pth"
    
    # ---------------------------------
    
    # create model
    frame_dim = 50 # how many numbers are in each frame?
    m = MotionModel(input_dim=frame_dim, output_dim=frame_dim)
    m = m.to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)
    
    # train
   
    if TRAIN:
        # Generate a random seed
        global train_seed
        train_seed = random.randint(1, 100000)
        print(f'Training model {train_seed}...')
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')  # Initialize with infinity, so first instance is saved
    
        
        for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch"):
            # get sample batch of data
            xb,yb,mask = get_batch('train',data)
            # evaluate loss
            logits, loss = m(xb,yb, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if epoch % EVAL_EVERY == 0:
                losses = estimate_loss()
                print(f"\nTrain loss: {losses['train']:.6f} val loss: {losses['val']:.6f}")
                
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
    xb,yb,mask = get_batch('test',data)

    generated = m.generate(xb, FRAMES_GENERATE)
    unnorm_out = unnormalise_list(generated, max_x, min_x, max_y, min_y)
    
    # visualise and save
    for batch in unnorm_out:
        visualise_skeleton(batch, max_x, max_y, max_frames=FRAMES_GENERATE,save = True,save_path=None,prefix=f'adam_{EPOCHS}steps_optimised_6layer_dropout_lerp')
        
    print('Done!')