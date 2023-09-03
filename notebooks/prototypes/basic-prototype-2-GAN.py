#!/usr/bin/env python
# coding: utf-8

# # Basic Prototype 2
# 
# Builds on Basic Prototype 1, adds yolov8 pose detection. Landmarks will be saved in a datastructure which then will be visualised and probably exported as a JSON so that it can be trained on Google Colab. This prototype focuses on testing if an AI model can be trained to interpolate and output the correct landmarks to continue the video sequence after being trained on the sequences of the videos.
# 
# Pose detection wise, Openpose is better since it has more landmarks and also tracks the foot, it is also based on C++ so it's quicker, but I cannot get openpose to work yet, so I will implement using YOLOv8 for now just to test the AI model structure. 
# 
# Mediapipe is also a possibility but it can only detect 1 person at a time and is much slower. 
# 
# Still the broadcast on OBS has to be started manually first.

# In[1]:


# imports
from memory_profiler import profile


@profile(precision=4)
def main():
    import pytchat
    import cv2
    import glob
    import ultralytics
    import torch
    import time
    import numpy as np
    from collections import defaultdict
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    import mdn
    import random
    import tensorflow as tf



    files = glob.glob('G:/UAL_Thesis/raw_videos/*')
    print(files)


    # In[2]:


    from PIL import Image

    def draw_skeleton(result):
        """draw frame from YOLOv8 results"""
        for r in result:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        return np.array(im)[..., ::-1]  # Convert PIL Image back to BGR numpy array


    # In[3]:


    # test YOLOv8 pose recognition with 1 file first

    """Check hardware and load model"""


    # Check if GPU is available otherwise use CPU for torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # load model
    video_path = files[1]
    model = ultralytics.YOLO('yolov8n-pose.pt')


    # If GPU is available set model to use half-precision floating-point numbers
    if torch.cuda.is_available():
        model.half().to(device)

    # predict

    cap = cv2.VideoCapture(video_path)
    start_time = time.time()

    # resize cv2 window

    # Get the original video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the desired width and calculate the height to maintain the aspect ratio
    desired_width = 540  # You can change this value
    aspect_ratio = height / width
    desired_height = int(desired_width * aspect_ratio)

    # Get video, pose analyse and display pose detection frame by frame
    pose_results = defaultdict(int)
    i=0

    while(cap.isOpened() and time.time()):
        ret, frame = cap.read()
        if ret == True:
            result = model.predict(frame)
            pose_results[i] = result
            i = i+1
            annotated_frame = draw_skeleton(result)
            # Resize the frame while maintaining the aspect ratio
            resized_frame = cv2.resize(annotated_frame, (desired_width, desired_height))
            # comment out because using too much RAM - run headless
            # cv2.imshow('Frame', resized_frame)

            # # Press Q on keyboard to exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
        else:
            break

    cap.release()
    cv2.destroyAllWindows




    # In[4]:


    def define_keypoints():
        return {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16
    }


    # In[5]:


    from collections import defaultdict

    # LSTM - adapted from AI for media
    # https://git.arts.ac.uk/tbroad/AI-4-Media-22-23/blob/main/Week%205.1%20LSTM%20for%20forecasting%20and%20movement%20generation/Generating_Movement_Sequences_with_LSTM.ipynb

    # define keypoints
    keypoints = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16
    }


    # convert tensor of xy coordinate per frame into a pandas time series
    keypoints_dict_master = defaultdict(dict)

    for frame, results in pose_results.items():
        for idx, person in enumerate(results):
            # Check if person has the attribute 'keypoints' and it has the attribute 'xy'
            if hasattr(person, 'keypoints') and hasattr(person.keypoints, 'xy'):
                tensor_values = person.keypoints.xy

                # Convert tensor to dictionary format
                keypoint_coordinates = {key: tensor_values[0][value] for key, value in keypoints.items() if value < len(tensor_values[0])}

                # Use idx as a unique identifier for each person
                keypoints_dict_master[frame][idx] = keypoint_coordinates


    # # GAN - Temporal GANs

    # In[6]:


    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    # https://github.com/pfnet-research/tgan2
    # prototyping with this one for now because it seems easy
    # https://github.com/amunozgarza/tsb-gan


    # In[7]:


    # actually going to just try adapt a simple DCGAN from pytorch tutorials first because I'm super confused
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

    #%matplotlib inline
    import argparse
    import os
    import random
    import torch
    import torch.nn as nn
    import torch.nn.parallel
    import torch.optim as optim
    import torch.utils.data
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import torchvision.utils as vutils
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results


    # In[8]:


    # Data preparation
    # convert to pandas of each keypoint coordinate for each frame

    # Flatten the dictionary
    rows = []
    for frame, persons in keypoints_dict_master.items():
        for person, keypoints in persons.items():
            row = {'frame': frame, 'person': person}
            for keypoint, coordinates in keypoints.items():
                row[f'{keypoint}_x'] = coordinates[0]
                row[f'{keypoint}_y'] = coordinates[1]
            rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    df


    # In[9]:


    from sklearn.preprocessing import MinMaxScaler
    import torch
    import traceback
    from tqdm import tqdm

    global HEIGHT, WIDTH,nc
    HEIGHT = 256
    WIDTH = 256
    nc = 3  #number of channels

    # Functions for slicing up data
    def slice_sequence_examples(sequence, num_steps):
        xs = []
        for i in range(len(sequence) - num_steps - 1):
            example = sequence[i: i + num_steps]
            xs.append(example)

            # output is list of list of num_steps number of rows (e.g. num_setps =  50 will be 50 first rows, all columns)
        return xs

    def seq_to_singleton_format(examples):
        # Takes the sliced sequences and separates each sequence into input (all elements except the last one) and output (just the last element).
        # up until last sequence used as primer

        xs = []
        ys = []
        for ex in examples:
            xs.append(ex[:-1])
            ys.append(ex[-1])
        return (xs,ys)

    def keypoints_to_image(scaled_keypoints_df):
        colors = {
        "nose": (255, 0, 0),        # Red
        "left_eye": (0, 255, 0),    # Green
        "right_eye": (0, 0, 255),   # Blue
        "left_ear": (255, 255, 0),  # Yellow
        "right_ear": (255, 0, 255), # Magenta
        "left_shoulder": (0, 255, 255),  # Cyan
        "right_shoulder": (255, 165, 0), # Orange
        "left_elbow": (255, 69, 0),     # Red-Orange
        "right_elbow": (0, 128, 0),     # Green (Lime)
        "left_wrist": (255, 20, 147),   # Deep Pink
        "right_wrist": (255, 140, 0),   # Dark Orange
        "left_hip": (0, 128, 128),      # Teal
        "right_hip": (255, 99, 71),     # Tomato
        "left_knee": (0, 255, 0),       # Green (using a different shade)
        "right_knee": (255, 69, 0),     # Red-Orange (using a different shade)
        "left_ankle": (0, 255, 255),    # Cyan (using a different shade)
        "right_ankle": (255, 165, 0)    # Orange (using a different shade)
    }
        # data frame coloumns are in the format of keypoint_x and keypoint_y
        # takes keypoints dataframe and returns a list of frames of keypoints, each keypoint body part is drawn in a different colour according to dict
        # on black (0 pad) background - turn to 4D

        frames = []

        for i,row in tqdm(scaled_keypoints_df.iterrows()):
            # match column name to keypoint name
            # black background
            img = np.zeros((HEIGHT,WIDTH,3), np.uint8)
            for keypoint in keypoints.keys():

                # draw circle at x,y coordinates
                try:
                    cv2.circle(img, (int(float(row[f"{keypoint}_x"])*WIDTH), int(float(row[f"{keypoint}_y"])*HEIGHT)), 5, colors[keypoint], -1)
                except Exception as e:
                    print(i)
                    print(keypoint)
                    traceback.print_exc()
                    continue

            frames.append(img)

        return frames




    # Normalising our data with min max
    sc = MinMaxScaler()
    scaled = sc.fit_transform(df.values)
    scaled_df= pd.DataFrame(scaled, columns=df.columns)

    # converting df to images
    frames = keypoints_to_image(scaled_df)

    # Turning our dataframe structure into an array, excluding the first 2 columns of person and frame
    seq = np.array(frames)


    # Defining and using our window size to create our inputs X and outputs y
    SEQ_LEN = 50
    slices = slice_sequence_examples(seq, SEQ_LEN+1)
    X, y = seq_to_singleton_format(slices)

    X = np.array(X)
    y = np.array(y)

    print("Number of training examples:")
    print("X:", X.shape)
    print("y:", y.shape)


    # X: (number of frames, sequence length, 17 coordinates, x and y so 34)
    # y: (number of frames, 17 coordinates, x and y so 34) - no sequence length cause generative


    # In[10]:


    np.shape(X[-1])


    # In[11]:


    # BatchSize×Depth×Height×Width×Channels

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 8

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = HEIGHT

    # Number of frames for seed
    nf = 50

    # Size of z latent vector (i.e. size of genaerator input) (was 100)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 5

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1


    # In[12]:


    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    # Create the dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


    # In[13]:


    # custom weights initialization called on ``netG`` and ``netD``
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    # In[14]:


    import torch.nn as nn

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(

                # Input shape: [batch, nz, 1, 1, 1]
                # Output shape: [batch, ngf * 8, 4, 4, 4]
                # [16, 512, 2, 2, 4]
                nn.ConvTranspose3d(nz, ngf * 8, (2, 2, 4), (1, 1, 2), 0, bias=False),
                nn.BatchNorm3d(ngf * 8),
                nn.ReLU(True),

                # Output shape: [batch, ngf * 4, 8, 8, 10]
                # [16, 256, 4, 4, 12]
                nn.ConvTranspose3d(ngf * 8, ngf * 4, (2, 2, 6), (2, 2, 2), 0, bias=False),
                nn.BatchNorm3d(ngf * 4),
                nn.ReLU(True),

                # Output shape: [batch, ngf * 2, 16, 16, 15]
                # [16, 128, 8, 8, 17]
                nn.ConvTranspose3d(ngf * 4, ngf * 2, (2, 2, 6), (2, 2, 1), 0, bias=False),
                nn.BatchNorm3d(ngf * 2),
                nn.ReLU(True),

                # Output shape: [batch, ngf, 32, 32, 20]
                # [16, 64, 16, 16, 22]
                nn.ConvTranspose3d(ngf * 2, ngf, (2, 2, 6), (2, 2, 1), 0, bias=False),
                nn.BatchNorm3d(ngf),
                nn.ReLU(True),

                # Output shape: [batch, ngf, 64, 64, 22]
                # [16, 64, 32, 32, 24]
                nn.ConvTranspose3d(ngf, ngf, (2, 2, 3), (2, 2, 1), 0, bias=False),
                nn.BatchNorm3d(ngf),
                nn.ReLU(True),

                # Output shape: [batch, ngf, 128, 128, 46]
                # [16, 64, 64, 64, 50]
                nn.ConvTranspose3d(ngf, ngf, (2, 2, 4), (2, 2, 2), 0, bias=False),
                nn.BatchNorm3d(ngf),
                nn.ReLU(True),

                # Output shape: [batch, ngf, 128, 128, 46]
                # [16, 3, 128, 128, 50]
                nn.ConvTranspose3d(ngf, ngf, (2, 2, 1), (2, 2, 1), 0, bias=False),
                nn.BatchNorm3d(ngf),
                nn.ReLU(True),


                # Desired Output shape: [batch, nc, 256, 256, 50]
                # [16, 3, 256, 256, 50]
                nn.ConvTranspose3d(ngf, nc, (2, 2, 1), (2, 2, 1), 0, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            x = input
            print(x.shape)
            for layer in self.main:
                print(layer)
                print(f"before: {x.shape}")
                x = layer(x)
                print(f"after: {x.shape}")
            return x




    # In[15]:


    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)


    # In[16]:


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # Input is [64, 50, 256, 256, 3]
                # kernel size 4,4,4 - 4 frames 4x4 pixels
                nn.Conv3d(in_channels=nc, out_channels=ndf, kernel_size= (4, 4, 4), stride=(1, 2, 2), padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # torch.Size([64, 64, 128, 128, 1])
                nn.Conv3d(ndf, ndf*2, (4, 4, 4), (2, 2, 2), padding = (1,1,1), bias=False),
                nn.BatchNorm3d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ndf * 2, ndf * 4, (4, 4, 4), (2, 2, 2), 1, bias=False),
                nn.BatchNorm3d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ndf * 4, ndf * 8, (4, 4, 4), (2, 2, 2), 1, bias=False),
                nn.BatchNorm3d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ndf * 8, ndf*16, (3, 4, 4), (2, 2, 2), 1, bias=False),
                nn.Flatten(),
                # Flattened Size=Channels×Depth×Height×Width
                nn.Linear(1024*3*8*8, 1),  # Adjusted the input size to the linear layer
                nn.Sigmoid()
            )
        def forward(self, input):
            x = input
            print(x.shape)
            for layer in self.main:
                print(layer)
                print(f"before: {x.shape}")
                x = layer(x)
                print(f"after: {x.shape}")
            return x


    # In[17]:


    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)


    # In[18]:


    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(batch_size, nz, 1, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    # In[19]:


    for i,data in enumerate(dataloader, 0):
        print(i)
        print(data[0].shape)
        print(data[1].shape)



    # In[1]:


    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            print(i)

            try:
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                # batch, depth, height, width, rgb
                real_cpu_a = data[0].float().to(device)
                # batch, rgb, depth, height, width
                real_cpu = real_cpu_a.permute(0, 4, 1, 2, 3)
                # batch size
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                print('Discriminator')
                # batch, rgb, depth,height, width
                output = netD(real_cpu).view(-1)
                # Calculate loss on all-real batch bvNC|
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, 1, device=device)
                # Generate fake image batch with G
                print('Generator')
                # batch, rgb, height, width, depth
                fake = netG(noise)
                # batch, rgb, depth,height, width
                fake = fake.permute(0, 1, 4, 2, 3)
                label.fill_(fake_label)
                # Classify all fake batch with D
                print('Classify all fake batch with D')
                 # batch, rgb, depth,height, width
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, num_epochs, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                        fake_sliced = fake[:, :, :, :, 0]  # takes the first frame from each video
                    img_list.append(vutils.make_grid(fake_sliced, padding=2, normalize=True))

                iters += 1

            except RuntimeError as e:

                # visualise current frame

                print(np.shape(real_cpu))
                plt.imshow(np.transpose(vutils.make_grid(real_cpu, padding=2, normalize=True).cpu(),(1,2,0)))
                traceback.print_exc()


if __name__ == "__main__":
    main()