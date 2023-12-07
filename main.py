import multiprocessing
import queue
import time
import keyboard
import queue
from model import *
import pytchat
from data import *
from visuals import visualise_body, global_load_images
import pyglet

# Use a pipeline as a high-level helper
from transformers import pipeline
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Functions


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
    notes = f"""main"""
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

pipe = pipeline("text-classification", model="michellejieli/emotion_text_classifier")


tokenizer = AutoTokenizer.from_pretrained("michellejieli/emotion_text_classifier")
model = AutoModelForSequenceClassification.from_pretrained("michellejieli/emotion_text_classifier")
    
# initialise model------------------------------------------------------------
args = argparse.Namespace(
        BATCH_SIZE=8,
        BLOCK_SIZE=16,
        DROPOUT=0.2,
        LEARNING_RATE=0.0001,
        EPOCHS=30000,
        FRAMES_GENERATE=30,
        TRAIN=False,
        EVAL_EVERY=1000,
        CHECKPOINT_PATH="checkpoints/proto10_checkpoint.pth",
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
        notes = f"""main.py"""
    )
# Initialising ------------------------------------------------------------
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

    
def normalise_generated(unnorm_out, max_x, min_x, max_y, min_y, max_dx, min_dx, max_dy, min_dy,scale=1.5): 
    norm_out = []
    
    max_x = max_x * scale
    min_x = min_x / scale
    max_y = max_y * scale
    min_y = min_y / scale
    
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




# different from normal emotion labels - matches the sentiment analyser
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

emotion_data = {emotion: {"score": 0.0, "count": 0} for emotion in emotion_labels}

chat = pytchat.create(video_id="gCNeDWCI0vo")
terminate_threads = False

# This queue will hold the batches ready for visualization
viz_queue = queue.Queue()

def process_chat_message(c,shared_data):
    """Process a chat message and update emotion scores."""
    detected_emotion = None

    if c.message.startswith('!GALLERY INPUT!:'):  # Gallery input format: !GALLERY INPUT!:emotion=score
        parts = c.message.split('!GALLERY INPUT!:')[1].split('=')
        if len(parts) == 2:
            detected_emotion, score = parts
            try:
                score = float(score)
                emotion_data[detected_emotion]["score"] = min(1, score)
                emotion_data[detected_emotion]["count"] = 0
            except ValueError:
                print("Invalid score format.")
        else:
            print("Invalid gallery input format.")

    else:
        print(f"{c.datetime} [{c.author.name}]- {c.message}")
        result = pipe(c.message)  # Assuming pipe() returns emotion prediction
        print(result)

        if result:
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

    print("Updated average scores in shared_data:", shared_data['average_scores'])
    # print("Average scores:", shared_data['average_scores'])


# Batch generation function
def generate_new_batch(shared_data,last_frame=None):
    """Generate a new batch based on the current average scores."""
    # If initial_data is None or empty, initialize with default values
    init_flag= False
    if last_frame is None:
        print('LAST FRAME IS NONE')
        last_frame = torch.randn(1,5, 50).to(device)  # initialise with noise
        init_flag = True # First Frame

    last_frames = last_frame[0][-3:]
    norm_last_frames = normalise_generated(last_frames, max_x, min_x, max_y, min_y, max_x, min_x, max_y, min_y)
    new_input = torch.tensor([norm_last_frames]).to(device).float()
    print(shared_data['average_scores'])
    emotion_in = torch.tensor([shared_data['average_scores']]).to(device).float()

    # Generate the new frames
    generated_keypoints, generated_emotion = m.generate(new_input, emotion_in, FRAMES_GENERATE)
    
    detached_keypoints = generated_keypoints.detach().cpu()
    detached_emotion = generated_emotion.detach().cpu()
    
    emotion_vectors = (emotion_in, detached_emotion)
    
    # Example Usage
    max_movement = 100  # Maximum allowed movement per step
    max_length = 300
    unnorm_out =unnormalise_list_2D(detached_keypoints, max_x, min_x, max_y, min_y, max_x, min_x, max_y, min_y)
    if init_flag == False:
        # not first frame
        smoothed_keypoints = smooth_generated_sequence_with_cap(torch.tensor(unnorm_out, device=device), max_movement, max_length)
        # print(smoothed_keypoints)
        # print('smooth')
        # print(unnorm_out)
        # print('unormout')
    else:
        # first frame
        smoothed_keypoints = unnorm_out
        init_flag= False
        
    
    # print(init_flag)
    return smoothed_keypoints, emotion_vectors

def generate_batches_periodically(queue, shared_data,period=2, last_frames=None):
    while True:
        time.sleep(period)
        unnorm_out, emotion_vectors = generate_new_batch(shared_data, last_frames)
        print('GENERATED BATCH PUTTING IN QUEUE')
        for frame in tqdm(unnorm_out[0]):
            queue.put((frame, emotion_vectors))
        # print(last_frames)
        last_frames = unnorm_out
        
# Function to update the visualisation
def clear_sprites():
    global limb_sprites
    # delete any previous sprites-------------------------------------------------------------------------------------------------------------------
    try:
        # Delete existing sprites before redefining limb_sprites
        for sprite in limb_sprites.values():
            sprite.delete()  # This deletes the sprite from the GPU
        limb_sprites = {}

    except NameError:
        # print('Name error')
        pass


def update(dt):
    global frame_index
    # Function called at regular intervals by the pyglet clock
    if not viz_queue.empty():
        try:
            # Get a single frame from the queue and visualize it
            frame_data, emotion_vectors = viz_queue.get_nowait()
            # print('VISUALISING')
            visualise_body(frame_data, emotion_vectors, max_x, max_y, window,start_time,frame_index)  # Visualize it
            frame_index += 1
            # print(frame_index)
            
        except queue.Empty:
            pass


    # Check for keyboard interrupt (e.g., ESC key to exit)
    if keyboard.is_pressed('esc'):
        pyglet.app.exit()
        
# Function to process chat messages in a separate process
def chat_process(terminate_event):
    chat = pytchat.create(video_id="4c7_urOJnZI") # CHANGE LINK HERE
    while not terminate_event.is_set():
        for c in chat.get().sync_items():
            process_chat_message(c)  # Make sure this function uses shared_data appropriately
            # Update viz_queue or other shared resources as needed

from multiprocessing import Manager, shared_memory

if __name__ == '__main__':

    manager = Manager()
    shared_data = manager.dict()
    shared_data['average_scores'] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    
    # Shared event to signal termination
    terminate_event = multiprocessing.Event()
    
    # Create the Pyglet window
    window = pyglet.window.Window(int(max_x) + 50, int(max_y) + 50)
    global_load_images()
    print('WINDOW CREATED')
    
    # Set up pyglet clock to call visualisation's update function periodically
    pyglet.clock.schedule_interval(update, 0.15)  # Adjust interval as needed

    # Process communication queue
    viz_queue = multiprocessing.Queue()
    
    # Start the processes
    generation_process = multiprocessing.Process(target=generate_batches_periodically, args=(viz_queue, shared_data,2))
    generation_process.start()

    # Start chat processing process
    chat_proc = multiprocessing.Process(target=chat_process, args=(terminate_event,shared_data))
    chat_proc.start()


    # Required in global scope ----------------------------------------

    global start_time
    start_time = time.time()

    global frame_index
    frame_index = 0
    # Main --------------------------------------------------------------
    # Run pyglet app
    pyglet.app.run()

    # Clean up
    # Signal the chat process to terminate and wait for it to finish
    terminate_event.set()
    chat_proc.join()
    generation_process.join()
    pyglet.app.exit()

    # Clean up
    # Signal the chat process to terminate and wait for it to finish
    terminate_event.set()
    chat_proc.join()
    generation_process.join()
    pyglet.app.exit()