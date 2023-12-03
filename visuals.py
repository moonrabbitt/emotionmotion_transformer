import pyglet
import math
import time
from pyglet.shapes import Circle
import json
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.gl import *
from pyglet.graphics import Group
import os
from pyglet.text import Label
import random
import glob
import glsl
from memory_profiler import profile

# Set root directory
root_dir = "C:\\Users\\avika\\OneDrive\\Documents\\UAL\\interactive_dance_thesis"
os.chdir(root_dir)

# https://github.com/pyglet/pyglet/blob/master/examples/opengl/opengl_shader.py
class RenderGroup(Group):
    """A Group that enables and binds a Texture and ShaderProgram.

    RenderGroups are equal if their Texture and ShaderProgram
    are equal.
    """
    def __init__(self, texture, program, order=0, parent=None):
        """Create a RenderGroup.

        :Parameters:
            `texture` : `~pyglet.image.Texture`
                Texture to bind.
            `program` : `~pyglet.graphics.shader.ShaderProgram`
                ShaderProgram to use.
            `order` : int
                Change the order to render above or below other Groups.
            `parent` : `~pyglet.graphics.Group`
                Parent group.
        """
        super().__init__(order, parent)
        self.texture = texture
        self.program = program

    def set_state(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(self.texture.target, self.texture.id)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.program.use()

    def unset_state(self):
        glDisable(GL_BLEND)

    def __hash__(self):
        return hash((self.texture.target, self.texture.id, self.order, self.parent, self.program))

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.texture.target == other.texture.target and
                self.texture.id == other.texture.id and
                self.order == other.order and
                self.program == other.program and
                self.parent == other.parent)

def create_quad(x, y, texture):
    x2 = x + texture.width
    y2 = y + texture.height
    return x, y, x2, y, x2, y2, x, y2


# Constants for blinking
BLINK_INTERVAL = 10  # The number of frames before a blink happens
BLINK_DURATION = 2   # The number of frames the blink lasts



# Function to load and compile shaders
def load_shader(shader_file):
    with open(shader_file, 'r') as file:
        shader_source = file.read()
    return pyglet.graphics.shader.Shader(shader_source, 'fragment')




def return_properties(emotion_vector, connection):
    try:
        emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sad', 'Surprise']

        # Ensure the emotion_vector is the same length as emotion_labels
        if len(emotion_vector) != len(emotion_labels):
            raise ValueError("Length of emotion_vector must match the number of emotion_labels")

        # Choose an emotion based on the normalized dominance values (probabilities)
        chosen_emotion = random.choices(emotion_labels, weights=emotion_vector, k=1)[0]

        # Join the connection names into a single string
        connection = '_'.join(connection)

        # Construct the file path
        file_path = f'D:\\Interactive Dance Thesis Tests\\visualisations\\{chosen_emotion}_NORM\\{connection}\\*.png'
        path = random.choice(glob.glob(file_path))

        scale = 1.0
        if chosen_emotion == 'Happiness':
            scale = 1.0
        elif chosen_emotion == 'Sad':
            scale = 1.5
        elif chosen_emotion == 'Surprise':
            scale = 1.0
        elif chosen_emotion == 'Disgust':
            scale = 1.2
        else:
            scale = 1.0


        return path, scale

    except IndexError:
        # body part not found
        return None, None

def global_load_images():
    # GPU memory management - store path to not load sprite every frame
    global loaded_images
    loaded_images = {}  # Dictionary to store loaded images

# @profile
def visualise_body(frame_data, emotion_vectors, max_x, max_y,window,start_time,frame_index):
    # print('visualising body')
    # clear memory
    global limb_sprites

    clear_sprites()
    # Preprocess-------------------------------------------------------------------------------------------------------------------
    emotion_in, generated_emotion = emotion_vectors
    emotion_in = emotion_in[0].tolist()  # Assuming emotion_in is a tensor
    emotion_out = generated_emotion[0].tolist()  # Assuming generated_emotion is a tensor

    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sad', 'Surprise']

    # Get the index of the maximum value in emotion_out
    max_emotion_index = emotion_out.index(max(emotion_out))

    # Get the corresponding emotion label
    dominant_emotion = emotion_labels[max_emotion_index]



    # Load the shaders-------------------------------------------------------------------------------------------------------------------
    shader_program,program = glsl.create_program(dominant_emotion)



    # overlay_shader_program = ShaderProgram(vert_shader,frag_shader)
    # overlay_program =pyglet.graphics.shader.ComputeShaderProgram(_glitch_fragment_shader_source)




    # Load the keypoints mapping-------------------------------------------------------------------------------------------------------------------

    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho',
                    'L-Elb', 'L-Wr', 'MidHip', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear',
                    'L-Ear', 'L-BigToe', 'L-SmallToe', 'L-Heel', 'R-BigToe',
                    'R-SmallToe', 'R-Heel', 'R-Pupil' , 'L-Pupil','Head','L-Hand','R-Hand','Mouth'] #L-Pupil and R-Pupil are not in the original keypointsMapping

    # Define the limb connections using names
    limb_connections_names = [
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
        ("R-Ank", "R-Heel"),
        ("R-Sho", "L-Sho","L-Hip","R-Hip"),
        ("Head",),
        ("L-BigToe","L-SmallToe"),
        ("R-BigToe","R-SmallToe"),
        ('L-Hand',),
        ('R-Hand',),
        ('L-Eye',),
        ('L-Pupil',),
        ('R-Eye',),
        ('R-Pupil',),
        ('Mouth',),

    ]

    # Loading specific images for each limb connection
    limb_sprites = {}
    offset = 5
    _,scale = return_properties(emotion_out,('Head',))
    for connection in limb_connections_names:
        # print(connection)
        # print(len(connection))
        try:
            if len(connection) == 1:
                if connection == ('Head',):
                    image_path,scale = return_properties(emotion_out,connection)
                    if image_path not in loaded_images:
                        loaded_images[image_path] = pyglet.image.load(image_path)
                    limb_image = loaded_images[image_path]
                    limb_image.anchor_x = limb_image.width // 2
                    limb_image.anchor_y = limb_image.height//2
                    head_width = limb_image.width
                    head_height = limb_image.height
                elif connection == ('L-Hand',):
                    image_path,scale = return_properties(emotion_out,connection)
                    if image_path not in loaded_images:
                        loaded_images[image_path] = pyglet.image.load(image_path)
                    limb_image = loaded_images[image_path]
                    limb_image.anchor_x = limb_image.width // 2
                    limb_image.anchor_y = limb_image.height + offset
                    hand_width = limb_image.width
                    hand_height = limb_image.height
                elif connection == ('R-Hand',):
                    image_path,scale = return_properties(emotion_out,connection)
                    if image_path not in loaded_images:
                        loaded_images[image_path] = pyglet.image.load(image_path)
                    limb_image = loaded_images[image_path]
                    limb_image.anchor_x = limb_image.width // 2
                    limb_image.anchor_y = limb_image.height - offset
                    hand_width = limb_image.width
                    hand_height = limb_image.height

                elif connection == ('L-Eye',):
                    image_path,scale = return_properties(emotion_out,connection)
                    if image_path not in loaded_images:
                        loaded_images[image_path] = pyglet.image.load(image_path)
                    limb_image = loaded_images[image_path]
                    limb_image.anchor_x = 0
                    limb_image.anchor_y = limb_image.height//2
                    eye_width = limb_image.width
                    eye_height = limb_image.height

                elif connection == ('R-Pupil',):
                    # Load pupil images and set properties
                    image_path,scale = return_properties(emotion_out,connection)
                    if image_path not in loaded_images:
                        loaded_images[image_path] = pyglet.image.load(image_path)
                    limb_image = loaded_images[image_path]
                    limb_image.anchor_x = 0
                    limb_image.anchor_y = limb_image.height//2

                elif connection == ('R-Eye',):
                    image_path,scale = return_properties(emotion_out,connection)
                    if image_path not in loaded_images:
                        loaded_images[image_path] = pyglet.image.load(image_path)
                    limb_image = loaded_images[image_path]
                    limb_image.anchor_x = limb_image.width
                    limb_image.anchor_y = limb_image.height//2
                    eye_width = limb_image.width
                    eye_height = limb_image.height

                elif connection == ('L-Pupil',):
                    image_path,scale = return_properties(emotion_out,connection)
                    if image_path not in loaded_images:
                        loaded_images[image_path] = pyglet.image.load(image_path)
                    limb_image = loaded_images[image_path]
                    limb_image.anchor_x = limb_image.width
                    limb_image.anchor_y = limb_image.height//2

                elif connection == ('Mouth',):
                    image_path,scale = return_properties(emotion_out,connection)
                    if image_path not in loaded_images:
                        loaded_images[image_path] = pyglet.image.load(image_path)
                    limb_image = loaded_images[image_path]
                    limb_image.anchor_x = limb_image.width // 2
                    limb_image.anchor_y = limb_image.height


                limb_sprites[(keypointsMapping.index(connection[0]),)] = pyglet.sprite.Sprite(limb_image)

            elif len(connection) == 4:
                # Special case: multiple keypoints (e.g., four points)
                image_path,scale = return_properties(emotion_out,connection)
                if image_path not in loaded_images:
                    loaded_images[image_path] = pyglet.image.load(image_path)
                limb_image = loaded_images[image_path]
                limb_image.anchor_x = limb_image.width // 2
                limb_image.anchor_y = limb_image.height - offset
                body_width = limb_image.width
                body_height = limb_image.height

                limb_sprites[(keypointsMapping.index(connection[0]), keypointsMapping.index(connection[1]), keypointsMapping.index(connection[2]), keypointsMapping.index(connection[3]))]  = pyglet.sprite.Sprite(limb_image)

            else:
                # Standard case: connection between two keypoints
                try:
                    image_path,scale = return_properties(emotion_out,connection)
                    if image_path not in loaded_images:
                        loaded_images[image_path] = pyglet.image.load(image_path)
                    limb_image = loaded_images[image_path]
                    limb_image.anchor_x = limb_image.width // 2
                    limb_image.anchor_y = limb_image.height - offset

                    limb_sprites[(keypointsMapping.index(connection[0]), keypointsMapping.index(connection[1]))] = pyglet.sprite.Sprite(limb_image)

                except:
                    continue

        except:
            continue


    # Function to draw each frame
    def draw_frame(frame_data):

        is_blinking = frame_index % BLINK_INTERVAL < BLINK_DURATION


        for i in range(25):  # Assuming 25 keypoints
            x = frame_data[i * 2]
            y = max_y - frame_data[i * 2 + 1]
            circle = Circle(x, y, 5, color=(255, 255, 255))
            circle.draw()


        for limb, sprite in limb_sprites.items():
            sprite.x = 0
            sprite.y = 0
            sprite.rotation = 0
            sprite.scale = 1

            if len(limb) == 4:

                sprite.width = body_width
                sprite.height = body_height

                # Calculate the midpoints for shoulders and hips
                mid_sho_x = (frame_data[limb[0] * 2] + frame_data[limb[1] * 2]) / 2
                mid_sho_y = max_y -(frame_data[limb[0] * 2 + 1] + frame_data[limb[1] * 2 + 1]) / 2
                mid_hip_x = (frame_data[limb[2] * 2] + frame_data[limb[3] * 2]) / 2
                mid_hip_y = max_y -(frame_data[limb[2] * 2 + 1] + frame_data[limb[3] * 2 + 1]) / 2

                # Calculate the distances for height and width
                height = abs(mid_sho_y - mid_hip_y)
                # height = abs(mid_sho_y - mid_hip_y)
                width_sho = abs(frame_data[limb[0] * 2] - frame_data[limb[1] * 2])
                width_hip = abs(frame_data[limb[2] * 2] - frame_data[limb[3] * 2])
                width =  width_hip   # Add some padding
                # Set sprite properties

                sprite.scale_x = (width / sprite.width)*1.8 *scale
                sprite.scale_y = (height / sprite.height)*1.2 *scale

                sprite.rotation = math.atan2(mid_hip_y - mid_sho_y, mid_hip_x - mid_sho_x)

                # Set the position to the midpoint between shoulders and hips
                sprite.x, sprite.y = mid_sho_x , mid_sho_y

            elif len(limb) == 1:
                # get index of head in keypointsMapping


                if limb == (keypointsMapping.index('Head'),): #Head
                    sprite.width = head_width
                    sprite.height = head_height
                    sprite.scale = 0.3*scale
                    limb = (1,0) #neck,nose


                elif limb == (keypointsMapping.index('L-Eye'),): # L-Eye
                    sprite.width = eye_width
                    sprite.height = eye_height*2
                    sprite.scale = 0.02 *scale

                elif limb == (keypointsMapping.index('R-Eye'),): # R-Eye
                    sprite.width = eye_width
                    sprite.height = eye_height*2
                    sprite.scale = 0.02*scale

                elif limb == (keypointsMapping.index('L-Pupil'),) : # L-Pupil
                    try:
                        if not is_blinking:
                            sprite.width = eye_width
                            sprite.height = sprite.width
                            sprite.scale = 0.01*scale
                            limb = (15,)
                        else:
                            continue
                    except NameError:
                        sprite.scale = scale
                        limb = (15,)

                elif limb == (keypointsMapping.index('R-Pupil'),): # R-Pupil
                    try:
                        if not is_blinking:
                            sprite.width = eye_width
                            sprite.height = sprite.width
                            sprite.scale = 0.01*scale
                            limb = (16,)
                        else:
                            continue
                    except NameError:
                        sprite.scale = scale
                        limb = (16,)

                elif limb == (keypointsMapping.index('L-Hand'),): # L-Hand
                    sprite.width = hand_width
                    sprite.height = hand_height
                    sprite.scale = 0.3*scale
                    limb = (7,) # L-Wr

                elif limb == (keypointsMapping.index('R-Hand'),): # R-Hand
                    sprite.width = hand_width
                    sprite.height = hand_height
                    sprite.scale = 0.3*scale
                    limb = (4,) # R-Wr

                elif limb == (keypointsMapping.index('Mouth'),): # Mouth
                    sprite.width = hand_width*2
                    sprite.height = hand_height/2
                    sprite.scale = 0.3*scale
                    limb = (0,)

                else:
                    sprite.width = hand_width
                    sprite.height = hand_height
                    sprite.scale = 0.3*scale


                if len(limb) > 1:
                    # Find midpoint between neck and nose
                    mid_x = (frame_data[limb[0] * 2] + frame_data[limb[1] * 2]) / 2
                    mid_y = max_y - (frame_data[limb[0] * 2 + 1] + frame_data[limb[1] * 2 + 1]) / 2

                    # Calculate the 1/4 position below the nose
                    nose_y = max_y - frame_data[limb[1] * 2 + 1]
                    quarter_below_nose = nose_y + (nose_y - mid_y) / 4

                    # Adjust sprite position
                    sprite.x = mid_x
                    sprite.y = quarter_below_nose
                else:
                    sprite.x, sprite.y = frame_data[limb[0] * 2], max_y - frame_data[limb[0] * 2 + 1]



            else:
                start_idx, end_idx = limb
                start_x, start_y = frame_data[start_idx * 2], max_y - frame_data[start_idx * 2 + 1]
                end_x, end_y = frame_data[end_idx * 2], max_y - frame_data[end_idx * 2 + 1]  # y needs to invert

                # # Calculate midpoint, angle, and distance
                angle = (math.atan2(start_y - end_y, end_x - start_x) -(math.pi/2))  # Flipped y-coordinates means we also change the order here
                distance = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

                # Set sprite properties

                sprite.rotation = math.degrees(angle)

                sprite.scale = (distance / sprite.height)*1.1 *scale

                sprite.x, sprite.y = start_x, start_y

            sprite.draw()



    # Pyglet draw event
    @window.event
    def on_draw():
        window.clear()

        background_batch = pyglet.graphics.Batch()

        # Background-----------------------------------------------------------------------------------------------
        args = glsl.return_args(dominant_emotion,start_time,window)
        
        glsl.shader_on_draw(dominant_emotion, shader_program, program, background_batch,window,args)

        background_batch.draw()

        # Sprite -----------------------------------------------------------------------------------------------

        draw_frame(frame_data)




        # -----------------------------------------------------------------------------------------------

        # Overlay effects -----------------------------------------------------------------------------------------------

        # foreground = pyglet.graphics.Batch()

        # overlay_tex = pyglet.image.Texture.create(window.width, window.height, internalformat=GL_RGBA32F)
        # overlay_tex.bind_image_texture(unit=overlay_program.uniforms['img_output'].location)

        # overlay_program['resolution'] = (window.width, window.height)
        # overlay_program['time'] = current_time

        # with overlay_program:
        #     overlay_program.dispatch(overlay_tex.width, overlay_tex.height, 1, barrier=GL_ALL_BARRIER_BITS)

        # overlay_group = RenderGroup(overlay_tex, overlay_shader_program)
        # indices = (0, 1, 2, 0, 2, 3)
        # vertex_positions = create_quad(0, 0, overlay_tex)

        # overlay_vertex_list = overlay_shader_program.vertex_list_indexed(4, GL_TRIANGLES, indices, foreground, overlay_group,
        #                                                 position=('f', vertex_positions),
        #                                                 tex_coords=('f', overlay_tex.tex_coords))

        # foreground.draw()


        # Load the emotion vectors-------------------------------------------------------------------------------------------------------------------

        if emotion_vectors is not None:

            emotion_in_percentages = [
                f"{int(e * 100)}% {emotion_labels[i]}"
                for i, e in enumerate(emotion_in) if round(e * 100) > 0
            ]

            emotion_out_percentages = [
                f"{int(e * 100)}% {emotion_labels[i]}"
                for i, e in enumerate(emotion_out) if round(e * 100) > 0
            ]

            y0, dy = 30, 15  # Starting y position and line gap

            # Draw emotion_in percentages
            for i, line in enumerate(emotion_in_percentages):
                y = window.height - y0 - i * dy
                label = Label(line, x=10, y=y, font_size=12, color=(255, 255, 255, 255))
                label.draw()

            # Draw emotion_out percentages
            for i, line in enumerate(emotion_out_percentages):
                y = window.height - y0 - i * dy
                label = Label(line, x=window.width - 120, y=y, font_size=12, color=(255, 255, 255, 255))
                label.draw()
        #---------------------------------------------------------------------------------------------------


def clear_sprites():
    global limb_sprites
    # delete any previous sprites-------------------------------------------------------------------------------------------------------------------
    try:
        # Delete existing sprites before redefining limb_sprites
        for sprite in limb_sprites.values():
            sprite.delete()  # This deletes the sprite from the GPU
        limb_sprites = {}

    except NameError:
        print('Name error')
        pass


if __name__ == '__main__':
    import queue
    import torch


    # Read data from a JSON file
    with open('data/data.json', 'r') as file:
        loaded_data = json.load(file)

    # Accessing the data
    unnorm_out = loaded_data["unnorm_out"]
    max_x = loaded_data["max_x"]
    max_y = loaded_data["max_y"]
    global window
    window = pyglet.window.Window(int(max_x) + 50, int(max_y) + 50)

    frame_queue = queue.Queue()  # Queue to hold individual frames

    # Preload frames into the queue
    for frame in unnorm_out:
        frame_queue.put(frame)
    print(len(unnorm_out))

    # emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sad', 'Surprise']

    # happy emotion vector for testing
    emotion_vectors = (torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0]]), torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0,0.0]]))
    
    global_load_images()

    def update(dt):
        global frame_index
        if not frame_queue.empty():
            frame_data = frame_queue.get()  # Get the next frame from the queue
            visualise_body(frame_data, emotion_vectors, max_x, max_y, window, start_time, frame_index)  # Visualize it

            frame_index += 1
            # print(frame_index)

        else:
            pyglet.app.exit()

    pyglet.clock.schedule_interval(update, 0.05)  # Adjust interval as needed

    # Required in global scope ----------------------------------------


    global frame_index
    frame_index = 0

    global start_time
    start_time = time.time()

    # Required in global scope ----------------------------------------

    pyglet.app.run()

