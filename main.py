import threading
import queue
import time
import keyboard
import queue
from model import *
import pytchat
from data import *
from visuals import visualise_body

# Use a pipeline as a high-level helper
from transformers import pipeline


pipe = pipeline("text-classification", model="michellejieli/emotion_text_classifier")
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("michellejieli/emotion_text_classifier")
model = AutoModelForSequenceClassification.from_pretrained("michellejieli/emotion_text_classifier")

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
    

# initialise model------------------------------------------------------------


args = argparse.Namespace(
        BATCH_SIZE=8,
        BLOCK_SIZE=16,
        DROPOUT=0.2,
        LEARNING_RATE=0.0001,
        EPOCHS=30000,
        FRAMES_GENERATE=300,
        TRAIN=False,
        EVAL_EVERY=1000,
        CHECKPOINT_PATH="checkpoints/proto9_checkpoint_emotion3.pth",
        L1_LAMBDA=None,
        L2_REG=0.0,
        FINETUNE=False,
        FINE_TUNING_LR=1e-5,
        FINE_TUNING_EPOCHS=100000,
        PENALTY=False,
        LATENT_VIS_EVERY=1000,
        USE_MDN = True,
        PATIENCE= 35, #multiple of EVAL_EVERY * 10 - no early stopping if patience =0
        DATASET = "all",
        
        # NOTES---------------------------------
        notes = f"""Proto8 - # Define  MDN loss scheduling parameters - 
         # Define  MDN loss scheduling parameters
         linear rate increase
            
        trying to adapt Pette et al 2019, addign latent visualisation and analysing latent space. Might be slow, maybe take this out when live.
        
        Added MDN to increase variance of output as Bishop et al 1994. and Alemi et al 2017.
        
        Scheduling MDN weight to increase over time for loss so that mse loss has better chance of converging first because otherwise MDN loss is overpowering it.
        Currently linear function but maybe change this to exponential. 
        
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
# If args are provided, use those; otherwise, parse from command line
if args is None:
    args = parse_args()

# Set the global variables based on args
set_globals(args)

# Set global variables

processed_data= prep_data(dataset=args.DATASET)
# global train_data,train_emotions, val_data, val_emotions, frame_dim, max_x, min_x, max_y, min_y, max_dx, min_dx, max_dy, min_dy, threshold
train_data, train_emotions, val_data, val_emotions, frame_dim, max_x, min_x, max_y, min_y, max_dx, min_dx, max_dy, min_dy, threshold = processed_data

# create model
# global m
m = MotionModel(input_dim=frame_dim, output_dim=frame_dim,emotion_dim=7, blocksize=args.BLOCK_SIZE, hidden_dim=512, n_layers=8, dropout=args.DROPOUT)
m = m.to(device)

optimizer = torch.optim.Adam(m.parameters(), lr=args.LEARNING_RATE, weight_decay=args.L2_REG)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
# Load the model
print('Loading model...')


m, optimizer, scheduler, epoch, loss, train_seed = load_checkpoint(m, optimizer, args.CHECKPOINT_PATH,scheduler)

try:

    if USE_MDN:
        print('MDN layer is used.')
        keypoints_loss, emotion_loss, mdn_loss = loss
        total_loss = keypoints_loss + emotion_loss + mdn_loss
    else:
        print('MDN layer is not used.')
        keypoints_loss, emotion_loss = loss
        total_loss = keypoints_loss + emotion_loss
    print(f"Model {train_seed} loaded from {CHECKPOINT_PATH} (epoch {epoch}, keypoints loss: {keypoints_loss:.6f}, emotion loss: {emotion_loss:.6f} , total loss: {total_loss:.6f})")

except TypeError:
    print(f"Model {train_seed} loaded from {CHECKPOINT_PATH} (epoch {epoch}, total loss: {loss:.6f})")

# Functions
def normalise_generated(unnorm_out, max_x, min_x, max_y, min_y, max_dx, min_dx, max_dy, min_dy): 
    norm_out = []
    
    for frame in unnorm_out:
        norm_frame = []
        
        # Normalize the first 50 values (absolute x and y coordinates)
        for i in range(0, 50, 2):
            unnormalized_x = frame[i]
            unnormalized_y = frame[i+1]
            
            norm_x = 2 * (unnormalized_x - min_x) / (max_x - min_x) - 1
            norm_y = 2 * (unnormalized_y - min_y) / (max_y - min_y) - 1
            
            norm_frame.extend([norm_x, norm_y])
        
        # Append the emotion encoding without normalizing
    
        norm_out.append(norm_frame)
        
    return norm_out


# Initial setup
shared_data = {
    'average_scores': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

# different from normal emotion labels - matches the sentiment analyser
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

emotion_data = {emotion: {"score": 0.0, "count": 0} for emotion in emotion_labels}

chat = pytchat.create(video_id="gCNeDWCI0vo")
FRAMES_GENERATE = 150
terminate_threads = False

# This queue will hold the batches ready for visualization
viz_queue = queue.Queue()

def process_chat_message(c):
    """Process a chat message and update emotion scores."""
    print(f"{c.datetime} [{c.author.name}]- {c.message}")
    result = pipe(c.message)  # Assuming pipe() returns emotion prediction
    print(result)

    detected_emotion = result[0]['label']

    # Reset the counter for the detected emotion and boost its score
    emotion_data[detected_emotion]["count"] = 0
    score = result[0]['score']
    emotion_data[detected_emotion]["score"] = min(1, emotion_data[detected_emotion]["score"] + score)

    # Decay scores for other emotions and increase their counters
    for emotion, data in emotion_data.items():
        if emotion != detected_emotion:
            data["count"] += 1
            if data["count"] >= 5:
                data["score"] = 0
            else:
                data["score"] *= 0.5  # or any other decay factor you prefer

    # Normalize the scores so they add up to 1
    total_score = sum(data["score"] for data in emotion_data.values())
    if total_score > 0:
        for emotion in emotion_labels:
            emotion_data[emotion]["score"] /= total_score

    # Update average scores
    for i, emotion in enumerate(emotion_labels):
        shared_data['average_scores'][i] = emotion_data[emotion]["score"]

    print("Average scores:", shared_data['average_scores'])

# Batch generation function
def generate_new_batch(last_frame=None):
    """Generate a new batch based on the current average scores."""
    # If initial_data is None or empty, initialize with default values
    if last_frame is None:
        print('LAST FRAME IS NONE')
        last_frame = torch.randn(1,5, 50).to(device)  # initialise with noise

    last_frames = last_frame[0][-3:]
    norm_last_frames = normalise_generated(last_frames, max_x, min_x, max_y, min_y, max_x, min_x, max_y, min_y)
    new_input = torch.tensor([norm_last_frames]).to(device).float()
    emotion_in = torch.tensor([shared_data['average_scores']]).to(device).float()

    # Generate the new frames
    generated_keypoints, generated_emotion = m.generate(new_input, emotion_in, FRAMES_GENERATE)
    
    emotion_vectors = (emotion_in, generated_emotion)
    return unnormalise_list_2D(generated_keypoints, max_x, min_x, max_y, min_y, max_x, min_x, max_y, min_y), emotion_vectors

def generate_batches_periodically(period=2, last_frame=None):
    # initialise with last_frame = None
    while not terminate_threads:  
        time.sleep(period)
        unnorm_out, emotion_vectors = generate_new_batch(last_frame)
        viz_queue.put((unnorm_out, emotion_vectors))  
        last_frame = unnorm_out
        

def visualise(unnorm_out, emotion_vectors):
    # visualize
    emotion_in, generated_emotion = emotion_vectors 
    emotion_vectors = (emotion_in[0], generated_emotion[0]) #quick fix
    
    visualise_body(unnorm_out[0],max_x, max_y)
    # visualise_skeleton(unnorm_out[0], max_x, max_y, emotion_vectors,max_frames=FRAMES_GENERATE,save = False,save_path=None,prefix=f'{EPOCHS}_main_test',train_seed=train_seed,delta=False,destroy=False)

def visualise_batches():
    while not terminate_threads:  # Check the global termination flag
        batch = viz_queue.get()  # Get the tuple from the queue
        if batch is None:  # Check if the thread should terminate
            break
        unnorm_out, emotion_vectors = batch  # Unpack the tuple
        visualise(unnorm_out, emotion_vectors)

# Start the threads
visualisation_thread = threading.Thread(target=visualise_batches, daemon=True)
generation_thread = threading.Thread(target=generate_batches_periodically, args=(10,), daemon=True)

visualisation_thread.start()
generation_thread.start()


# Process chat messages
while chat.is_alive():
    if keyboard.is_pressed('esc'):  # Check if ESC key is pressed
        terminate_threads = True
        viz_queue.put(None)  # Put a None in the queue to signal the visualisation thread to terminate
        break  # Exit the main loop
    for c in chat.get().sync_items():
        process_chat_message(c)

cv2.destroyAllWindows()

# Wait for threads to finish if needed
visualisation_thread.join()
generation_thread.join()