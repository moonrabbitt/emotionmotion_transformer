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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data------------------------------------

def get_matched_danceDB_emotion(file):
    lowercase_file = file.lower()
    emotions_set = set(danceDB_emotions().keys())
    for emotion in emotions_set:
        if emotion.lower() in lowercase_file:
            return emotion
    return None

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

def preprocess_data(files: List[str] , dataset) -> dict:
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
        with open(file, 'r') as f:
            
            
            x=[]
            y=[]
            conf=[]
  
            # get different emotion code depending on dataset structure
            if dataset == "MEED":
                data = json.load(f)
                x = data['x']
                y = data['y']
                conf = data['confidence']
                emotion_code = file.split('_')[-2].split('\\')[0][3:-3]
                # 1 emotion per file - len(emotion) = len(files)
                emotions.append(emotion_labels_to_vectors(emotion_code))
            
            
            elif dataset == "DanceDB":
                data = json.loads(f.read())
                for i in range(len(data)-25):
                    nested_list = data[str(i)]
                    x.extend([coordinate[0] if coordinate is not None else 0 for coordinate in nested_list])
                    y.extend([coordinate[1] if coordinate is not None else 0 for coordinate in nested_list])
                conf = [1] * len(x)
                matched_emotion = get_matched_danceDB_emotion(file)
                # 1 emotion per file - len(emotion) = len(files)
                if matched_emotion is None:
                    matched_emotion = 'Mix' # default to mix if emotion not found - helps model less stuck
                emotions.append(encode_danceDB_emotion(matched_emotion))
              
                
                
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
    
    if validate_interpolation(x_list,y_list,files) == False:
        raise Exception('Interpolation not successful')

    # create deltas
    dx_list = delta_frames(x_list)
    dy_list = delta_frames(y_list)


    return {"x": x_list, "y": y_list,"dx": dx_list, "dy": dy_list, "confidence": conf_list, "emotions": emotions}

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


def delta_frames(vid_frames):
    """Find the difference between each frame (deltax deltay for all keypoints) for each video.
    use at point where x_list is [video[all x for all kps]]
    output: [first frame, delta frame, delta frame, ...]"""
    
    delta_vids = []
    for v, video in tqdm(enumerate(vid_frames)):
        delta_frames = []
        if len(video) % 25 != 0:
            raise Exception(f"Video {v} frames not divisible by 25 length: {len(video)}")
        
        # Iterate by skipping 25 keypoints (one full frame) at a time
        for i in range(0, len(video) - 25, 25):
            # Append the difference between the next 25 frames and the current 25 frames
            delta_frames.extend(np.subtract(video[i+25:i+50], video[i:i+25]))
        
        # Append zeros for the last frame
        delta_frames.extend([0] * 25)  
        delta_vids.append(delta_frames)
    
    return delta_vids


def add_delta_to_frames(input_frames, delta_frames):
    print("Adding deltas to frames...")
    kp_frames_with_delta = []
    
    # Ensure the outermost lists of input_frames and delta_frames have the same length
    if len(input_frames) != len(delta_frames):
        raise ValueError("Mismatched outer list sizes: {} and {}".format(len(input_frames), len(delta_frames)))
    
    # Iterate over paired (input_frame, delta_frame) elements from (input_frames, delta_frames)
    for input_frame, delta_frame in tqdm(zip(input_frames, delta_frames)):
        # Ensure the second-level lists of input_frame and delta_frame have the same length
        if len(input_frame) != len(delta_frame):
            raise ValueError("Mismatched second-level list sizes.")
        
        # Concatenate the innermost lists and append to kp_frames_with_delta
        new_frame = [in_f + del_f for in_f, del_f in zip(input_frame, delta_frame)]
        kp_frames_with_delta.append(new_frame)
    
    return kp_frames_with_delta


# Prepare data for training------------------------------------
def normalize_values_2D(frames, max_val = None,min_val=None):
    
    # normalise -1 to 1
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
    if max_val is None:
        max_val = max(flat_data)
        
    if min_val is None:
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


def validate_length(list_3d,length,message=None):
    print(f"Validating length of {message}")
    for i in range(len(list_3d)):
        for j in range(len(list_3d[i])):
            if len(list_3d[i][j]) != length:
                raise Exception(f"length of {i},{j} is {len(list_3d[i][j])}")
                return False
    return True

# Dealing with emotions------------------------------------
def emotion_labels_to_vectors(emotion_label):
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
    return label_to_vector[emotion_label] 

def emotion_to_encoding(emotion_label):
    """
    Convert an emotion label to its one-hot encoding.
    
    Parameters:
    - emotion_label (str): The label of the emotion.
    - emotion_labels (list of str): The list of all possible emotion labels.
    
    Returns:
    - list of int: The one-hot encoding of the emotion label.
    """
    
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sad', 'Surprise']

    encoding = [0] * len(emotion_labels)
    encoding[emotion_labels.index(emotion_label)] = 1
    return encoding


def add_emotions_to_frames(kp_frames, emotion_vectors):
    # kp_frames is normalised
    print("Adding emotions to frames...")
    kp_frames_with_emotion = []
    
    # length of emotion_vectors should be the same as kp_frames - 1 emotion per video
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
import random

def get_video_by_emotion(data, specific_emotion):
    """
    Retrieve a random video of a specified emotion from the dataset.

    Parameters:
    - data (list): The entire dataset, assumed to be a list of videos.
    - specific_emotion (str): The emotion label of the desired video.

    Returns:
    - list: A single video corresponding to the specified emotion.
    """
    # Convert the specific emotion to its encoding
    specific_emotion_tuple = tuple(emotion_to_encoding(specific_emotion))

    # Collect all videos with the specified emotion
    matching_videos = [video for video in data if tuple(video[0][-7:]) == specific_emotion_tuple]

    # If no video with the specified emotion is found, return None
    if not matching_videos:
        return None

    # Randomly select one video from the list of matching videos and return as tensor
    selected_video = random.choice(matching_videos)
    return torch.tensor([selected_video]).to(device).float()


def danceDB_emotions():
    # co pilot is really good at guessing emotions combinations?
    return {
        'Happy': {'Happiness': 1.0},
        'Miserable': {'Sad': 1.0},
        'Relaxed': {'Neutral': 0.9, 'Happiness': 0.1},
        'Sad': {'Sad': 1.0},
        'Satisfied': {'Happiness': 1.0},
        'Tired': {'Neutral': 1.0},
        'Excited': {'Happiness': 0.9, 'Surprise': 0.1},
        'Afraid': {'Fear': 1.0},
        'Angry': {'Anger': 1.0},
        'Annoyed': {'Anger': 0.5, 'Disgust': 0.5},
        'Bored': {'Neutral': 0.8, 'Sad': 0.1, 'Anger': 0.1},
        'Pleased': {'Happiness': 1.0},
        'Neutral': {'Neutral': 1.0},
        'Nervous': {'Fear': 0.5, 'Surprise': 0.5},
        'Mix': {'Anger': 0.143, 'Disgust': 0.143, 'Fear': 0.143, 'Happiness': 0.143, 'Neutral': 0.143, 'Sad': 0.143, 'Surprise': 0.143},
        'Curiosity': {'Surprise': 0.3,'Happiness': 0.3,'Neutral': 0.3,'Fear': 0.1}
    }

def encode_danceDB_emotion(emotion):
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sad', 'Surprise']
    
    emotion_mapping = danceDB_emotions()
    if emotion not in emotion_mapping:
        print(f"Emotion not found in mapping: {emotion}")
        return "Emotion not found in mapping"
    
    encoding = [0.0] * len(emotion_labels)
    for emotion_label, percentage in emotion_mapping[emotion].items():
        if emotion_label in emotion_labels:
            index = emotion_labels.index(emotion_label)
            encoding[index] = percentage

    return encoding

def add_noise_to_emotions(emotion_list,noise_level = 0.1):
    # This function adds noise to a list of emotion encodings
    # Each encoding should sum to 1 after noise addition
    
    def add_noise(encoding):
        # Generate noise that sums up to 10% of the total
        noise = np.random.random(len(encoding))
        noise *= noise_level / noise.sum()  # Scale the noise to sum up to 0.1
        noisy_encoding = encoding + noise
        # Normalize the encoding so that it sums to 1
        return noisy_encoding / noisy_encoding.sum()
    
    # Apply the add_noise function to each encoding in the list
    return np.array([add_noise(np.array(encoding)) for encoding in emotion_list])

# Programmatic functions------------------------------------

def stratified_split(data, emotions, test_size=0.1):
    # Organize data by class
    class_data = {}
    for video_index, _ in enumerate(data):
        # Convert the emotion list to a tuple to be used as a dictionary key
        emotion = tuple(emotions[video_index])
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
    train_emotions = [emotions[idx] for idx in train_indices]
    val_data = [data[idx] for idx in val_indices]
    val_emotions = [emotions[idx] for idx in val_indices]

    # Shuffle the train and val sets to ensure random order
    random.shuffle(train_data)
    random.shuffle(train_emotions)
    random.shuffle(val_data)
    random.shuffle(val_emotions)

    return (train_data, train_emotions), (val_data, val_emotions)



def shuffle_together(a, b):
    combined = list(zip(a, b))
    random.shuffle(combined)
    a[:], b[:] = zip(*combined)
    return a, b

def compute_threshold(dataset):
    """
    Compute the threshold for the magnitude of cumulative deltas based on the entire dataset.

    Parameters:
    - dataset (torch.Tensor): The dataset with shape [num_samples, sequence_length, feature_dim].

    Returns:
    - float: The computed threshold.
    """

    # Flatten the dataset to 2D structure
    flattened_dataset = [sample for batch in dataset for sample in batch]

    # Extract the deltas from the dataset and compute their magnitude
    deltas_magnitude = [[abs(x) for x in sample[50:100]] for sample in flattened_dataset]

    # Compute the cumulative magnitude of deltas over the entire sequence for each sample
    cumulative_deltas_magnitude = [sum(sample) for sample in deltas_magnitude]

    # Calculate the average cumulative magnitude of deltas across the entire dataset
    threshold = sum(cumulative_deltas_magnitude) / len(cumulative_deltas_magnitude)

    # Optionally, set the threshold slightly above the average
    threshold_multiplier = 1.2  # Increase threshold by 10%
    threshold *= threshold_multiplier

    return threshold

def get_meed_files()-> list:
    # direction = ['left', 'right', 'front']
    direction = ['front']
    meed_files = []
    for d in direction:
        meed_files.extend(glob.glob(f"G:/UAL_Thesis/affective_computing_datasets/multiview-emotional-expressions-dataset/*/{d}_*/processed_data.json"))
    return meed_files

def get_dance_db_files() -> list:
    emotions_set = {
        'Happy', 'Miserable', 'Relaxed', 'Sad', 'Satisfied', 'Tired', 
        'Excited', 'Afraid', 'Angry', 'Annoyed', 'Bored', 'Pleased',
        'Neutral', 'Nervous', 'Mix', 'Curiosity'
    }   
    path_pattern = "G:\\UAL_Thesis\\affective_computing_datasets\\DanceDBrenders\\DanceDB\\*\\*_keypoints.txt"
    all_files = glob.glob(path_pattern)
    # dance_db_files = [file for file in all_files for emotion in emotions_set if emotion.lower() in file.lower()]
    dance_db_files = all_files
    return dance_db_files

def prep_data(dataset):
    print(f"Preparing data for {dataset}...")
    
    if dataset == "MEED":
        meed_files = get_meed_files()
        processed_data =preprocess_data(meed_files, dataset)

    elif dataset == "DanceDB":
        dance_db_files = get_dance_db_files()
        processed_data = preprocess_data(dance_db_files, dataset)

    elif dataset == 'all':
        meed_files = get_meed_files()
        dance_db_files = get_dance_db_files()
        processed_data_MEED = preprocess_data(meed_files, 'MEED')
        processed_data_danceDB = preprocess_data(dance_db_files, 'DanceDB')
        processed_data = {key: processed_data_MEED[key] + processed_data_danceDB[key] for key in processed_data_MEED}
    
    x_list = processed_data['x']
    y_list = processed_data['y']
    dx_list = processed_data['dx']
    dy_list = processed_data['dy']
    emotion_vectors = processed_data['emotions']
    
    
    # prepare data for training - deltas from now on
    global max_x, min_x, max_y, min_y
    
    
    max_x, min_x, normalised_x = normalize_values_2D(x_list)
    max_y, min_y, normalised_y = normalize_values_2D(y_list)
    max_dx, min_dx, normalised_dx = normalize_values_2D(dx_list,max_x,min_x)
    max_dy, min_dy, normalised_dy = normalize_values_2D(dy_list,max_y,min_y)
    

    dkp_frames = create_kp_frames(normalised_dx, normalised_dy)  # 1D tensor array of 50 numbers (x,y,x,y --> 25 keypoints)
    kp_frames = create_kp_frames(normalised_x, normalised_y)  # 1D tensor array of 50 numbers (x,y,x,y --> 25 keypoints) - for getting first frames/validation
    
    validate_length(dkp_frames,50,message="dkp_frames")
    validate_length(kp_frames,50,message="kp_frames")
    
    # removed for proto9
    # data = add_delta_to_frames(kp_frames, dkp_frames)
    # validate_length(data,100,message="data after delta")
    
    data = kp_frames
    
    frame_dim = len(data[0][0]) # how many numbers are in each frame? - 50 kps xy + 50 deltas 
    print(f"frame_dim: {frame_dim}")
    
    global train_data, val_data
    
    (train_data, train_emotions), (val_data, val_emotions) = stratified_split(data, emotion_vectors, test_size=0.1)
    
    # add noise after stratified split
    train_emotions = add_noise_to_emotions(train_emotions)
    val_emotions = add_noise_to_emotions(val_emotions)
    
    # calculate threshold, maybe change this to entire data instead of just train
    threshold = compute_threshold(data)
    
    processed_data = (train_data, train_emotions, val_data, val_emotions, frame_dim, max_x, min_x, max_y, min_y, max_dx, min_dx, max_dy, min_dy, threshold)
 
    return processed_data

def cap_movements(frame1, frame2, max_movement):
    """
    Cap the movements between two frames.

    :param frame1: First frame tensor [C]
    :param frame2: Second frame tensor [C]
    :param max_movement: Maximum allowed movement
    :return: List of frames including capped intermediate frames
    """
    movement = frame2 - frame1
    max_movement_per_component = movement.abs().max()
    
    if max_movement_per_component <= max_movement:
        return [frame1, frame2]  # Movement is within the threshold

    # Calculate the number of steps needed
    num_steps = torch.ceil(max_movement_per_component / max_movement).int().item()
    
    # Generate intermediate frames
    interpolated_frames = [frame1]
    for step in range(1, num_steps):
        step_frame = frame1 + (movement * (step / num_steps))
        interpolated_frames.append(step_frame)

    interpolated_frames.append(frame2)
    return interpolated_frames

def pad_sequence_to_length(sequence, length):
    if len(sequence) < length:
        padding = torch.zeros((length - len(sequence), *sequence[0].shape)).to(device)
        sequence = torch.cat([sequence, padding])
    return sequence

def smooth_generated_sequence_with_cap(generated_sequence, max_movement):
    B, T, C = generated_sequence.shape
    print(f"Smoothing generated sequence with max movement {max_movement}...")
    
    smoothed_sequence = []
    max_length = 0
    for b in range(B):
        batch_sequence = [generated_sequence[b, 0]]
        print(f"Length before smoothing for batch {b}: {len(batch_sequence)}")  # Length of the sequence before smoothing
        for t in range(1, T):
            capped_frames = cap_movements(generated_sequence[b, t - 1], generated_sequence[b, t], max_movement)
            batch_sequence.extend(capped_frames[1:])  # Exclude the first frame to avoid duplicates
        max_length = max(max_length, len(batch_sequence))
        smoothed_sequence.append(torch.stack(batch_sequence))
        print(f"Length of sequence after smoothing for batch {b}: {len(batch_sequence)}")
    
    # Pad sequences to the same length
    padded_sequence = [pad_sequence_to_length(seq, max_length) for seq in smoothed_sequence]
    return torch.stack(padded_sequence).to(device)
 


if __name__ == "__main__":
    print("Running data.py as main")
    prep_data("all")
    print("Done!")