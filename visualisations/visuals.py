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

    return shader

def visualise_body(frame_data, emotion_vectors, max_x, max_y,window,start_time,frame_index):
    
    
    # Load the shaders-------------------------------------------------------------------------------------------------------------------
    
    
    _vertex_source = """#version 330 core
    in vec2 position;
    in vec3 tex_coords;
    out vec3 texture_coords;

    uniform WindowBlock 
    {                       // This UBO is defined on Window creation, and available
        mat4 projection;    // in all Shaders. You can modify these matrixes with the
        mat4 view;          // Window.view and Window.projection properties.
    } window;  

    void main()
    {
        gl_Position = window.projection * window.view * vec4(position, 1, 1);
        texture_coords = tex_coords;
    }
"""

    _fragment_source = """#version 330 core
    in vec3 texture_coords;
    out vec4 final_colors;

    uniform sampler2D our_texture;

    void main()
    {
        final_colors = texture(our_texture, texture_coords.xy);
    }
    """
    
    _compute_source = """#version 430 core
    layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

    layout(rgba32f) uniform image2D img_output;
    uniform float time;

    void main() {
    ivec2 texel_coord = ivec2(gl_GlobalInvocationID.xy);
    
    // Base color change on position
    float base_red = float(texel_coord.x) / (gl_NumWorkGroups.x);
    float base_green = float(texel_coord.y) / (gl_NumWorkGroups.y);

    // Modulate color based on time
    float time_red = (sin(time) + 1.0) / 2.0;  // Oscillates between 0 and 1
    float time_green = (cos(time) + 1.0) / 2.0;  // Oscillates between 0 and 1

    // Combine the position-based color with the time-based modulation
    vec4 value = vec4(base_red * time_red, base_green * time_green, 0.0, 1.0);

    imageStore(img_output, texel_coord, value);
}
    """


    vert_shader = Shader(_vertex_source, 'vertex')
    frag_shader = Shader(_fragment_source, 'fragment')
    shader_program = ShaderProgram(vert_shader, frag_shader)
    overlay_shader_program = ShaderProgram(vert_shader,frag_shader)
    program = pyglet.graphics.shader.ComputeShaderProgram(_compute_source)
    overlay_program =pyglet.graphics.shader.ComputeShaderProgram(_glitch_fragment_shader_source)
    

            

    # Load the keypoints mapping-------------------------------------------------------------------------------------------------------------------
    
    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                    'L-Elb', 'L-Wr', 'MidHip', 'R-Hip', 'R-Knee', 'R-Ank', 
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 
                    'L-Ear', 'L-BigToe', 'L-SmallToe', 'L-Heel', 'R-BigToe', 
                    'R-SmallToe', 'R-Heel', 'R-Pupil' , 'L-Pupil'] #L-Pupil and R-Pupil are not in the original keypointsMapping

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
        ("Nose",),
        ("L-BigToe","L-SmallToe"),
        ("R-BigToe","R-SmallToe"),
        ('L-Wr',),
        ('R-Wr',),
        ('L-Eye',),
        ('L-Pupil',),
        ('R-Eye',),
        ('R-Pupil',),
        
    ]
    
    # Loading specific images for each limb connection
    limb_sprites = {}
    for connection in limb_connections_names:
        # print(connection)
        # print(len(connection))        
        if len(connection) == 1:
            if connection == ('Nose',):
                image_path = f'G:/Downloads/happy/Nose_NORM.png'
                limb_image = pyglet.image.load(image_path)
                limb_image.anchor_x = limb_image.width // 2
                limb_image.anchor_y = limb_image.height//2
                head_width = limb_image.width
                head_height = limb_image.height
            elif connection == ('L-Wr',):
                image_path = f'G:/Downloads/happy/L-Wr_NORM.png'
                limb_image = pyglet.image.load(image_path)
                limb_image.anchor_x = limb_image.width // 2
                limb_image.anchor_y = limb_image.height
                hand_width = limb_image.width
                hand_height = limb_image.height
            elif connection == ('R-Wr',):
                image_path = f'G:/Downloads/happy/R-Wr_NORM.png'
                limb_image = pyglet.image.load(image_path)
                limb_image.anchor_x = limb_image.width // 2
                limb_image.anchor_y = limb_image.height
                hand_width = limb_image.width
                hand_height = limb_image.height
                
            elif connection == ('L-Eye',):
                image_path = f'G:/Downloads/happy/L-Eye_NORM.png'
                limb_image = pyglet.image.load(image_path)
                limb_image.anchor_x = 0
                limb_image.anchor_y = limb_image.height//2
                eye_width = limb_image.width
                eye_height = limb_image.height
            
            elif connection == ('L-Pupil',):
                # Load pupil images and set properties
                image_path = 'G:/Downloads/happy/L-Pupil_NORM.png' 
                limb_image = pyglet.image.load(image_path)
                limb_image.anchor_x = 0
                limb_image.anchor_y = limb_image.height//2
                
                
            elif connection == ('R-Eye',):
                image_path = f'G:/Downloads/happy/R-Eye_NORM.png'
                limb_image = pyglet.image.load(image_path)
                limb_image.anchor_x = limb_image.width 
                limb_image.anchor_y = limb_image.height//2
                eye_width = limb_image.width
                eye_height = limb_image.height
            
            elif connection == ('R-Pupil',):
                image_path = 'G:/Downloads/happy/R-Pupil_NORM.png'
                limb_image = pyglet.image.load(image_path)
                limb_image.anchor_x = limb_image.width 
                limb_image.anchor_y = limb_image.height//2
            
            limb_sprites[(keypointsMapping.index(connection[0]),)] = pyglet.sprite.Sprite(limb_image)
            
        elif len(connection) == 4:
            # Special case: multiple keypoints (e.g., four points)
            image_path = f'G:/Downloads/happy/R-Sho_L-Sho_L-Hip_R-Hip_NORM.png'
            limb_image = pyglet.image.load(image_path)
            limb_image.anchor_x = limb_image.width // 2
            limb_image.anchor_y = limb_image.height
            body_width = limb_image.width
            body_height = limb_image.height
            
            limb_sprites[(keypointsMapping.index(connection[0]), keypointsMapping.index(connection[1]), keypointsMapping.index(connection[2]), keypointsMapping.index(connection[3]))]  = pyglet.sprite.Sprite(limb_image)

        else:
            # Standard case: connection between two keypoints
            try:
                image_path = f'G:/Downloads/happy/{connection[0]}_{connection[1]}_NORM.png'
                limb_image = pyglet.image.load(image_path)
                limb_image.anchor_x = limb_image.width // 2
                limb_image.anchor_y = limb_image.height
                
                limb_sprites[(keypointsMapping.index(connection[0]), keypointsMapping.index(connection[1]))] = pyglet.sprite.Sprite(limb_image)
           
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
                
                sprite.scale_x = (width / sprite.width)*1.8
                sprite.scale_y = (height / sprite.height)*1.2
                
                sprite.rotation = math.atan2(mid_hip_y - mid_sho_y, mid_hip_x - mid_sho_x)
                
                # Set the position to the midpoint between shoulders and hips
                sprite.x, sprite.y = mid_sho_x , mid_sho_y 
            
            elif len(limb) == 1:
                if limb == (0,): # Nose
                    sprite.width = head_width
                    sprite.height = head_height
                    sprite.scale = 0.3
                
                elif limb == (15,): # L-Eye
                    sprite.width = eye_width
                    sprite.height = eye_height*2
                    sprite.scale = 0.02
                    
                elif limb == (16,): # R-Eye
                    sprite.width = eye_width
                    sprite.height = eye_height*2
                    sprite.scale = 0.02
                
                elif limb == (25,) : # L-Pupil
                    if not is_blinking:
                        sprite.width = eye_width
                        sprite.height = sprite.width
                        sprite.scale = 0.01
                        limb = (15,)
                    else:
                        continue
                
                elif limb == (26,): # R-Pupil
                    if not is_blinking:
                        sprite.width = eye_width
                        sprite.height = sprite.width
                        sprite.scale = 0.01
                        limb = (16,)
                    else:
                        continue
                    
                  
                else: # R-Wr
                    sprite.width = hand_width
                    sprite.height = hand_height
                    sprite.scale = 0.3
                    
                
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
                
                sprite.scale = (distance / sprite.height)*1.1
                
                sprite.x, sprite.y = start_x, start_y
            
            sprite.draw()


    # Pyglet draw event
    @window.event
    def on_draw():
        window.clear()
        background = pyglet.graphics.Batch()

        # Background-----------------------------------------------------------------------------------------------
        
        tex = pyglet.image.Texture.create(window.width, window.height, internalformat=GL_RGBA32F)
        tex.bind_image_texture(unit=program.uniforms['img_output'].location)
        
        current_time = (time.time()-start_time)*0.6
        program['time'] = current_time
        
        with program:
            program.dispatch(tex.width, tex.height, 1, barrier=GL_ALL_BARRIER_BITS)
        
        # tex = pyglet.resource.texture('data/leg.png')
        group = RenderGroup(tex, shader_program)
        indices = (0, 1, 2, 0, 2, 3)
        vertex_positions = create_quad(0, 0, tex)

        # count, mode, indices, batch, group, *data
        vertex_list = shader_program.vertex_list_indexed(4, GL_TRIANGLES, indices, background, group,
                                                        position=('f', vertex_positions),
                                                        tex_coords=('f', tex.tex_coords))
        
        background.draw()
        
        
        # Sprite -----------------------------------------------------------------------------------------------
        
        draw_frame(frame_data)
        
        
        # -----------------------------------------------------------------------------------------------
        
        
        # Overlay effects -----------------------------------------------------------------------------------------------
        
        foreground = pyglet.graphics.Batch()
        
        overlay_tex = pyglet.image.Texture.create(window.width, window.height, internalformat=GL_RGBA32F)
        overlay_tex.bind_image_texture(unit=overlay_program.uniforms['img_output'].location)
        
        overlay_program['resolution'] = (window.width, window.height)
        overlay_program['time'] = current_time
        
        with overlay_program:
            overlay_program.dispatch(overlay_tex.width, overlay_tex.height, 1, barrier=GL_ALL_BARRIER_BITS)
        
        overlay_group = RenderGroup(overlay_tex, overlay_shader_program)
        indices = (0, 1, 2, 0, 2, 3)
        vertex_positions = create_quad(0, 0, overlay_tex)
        
        overlay_vertex_list = overlay_shader_program.vertex_list_indexed(4, GL_TRIANGLES, indices, foreground, overlay_group,
                                                        position=('f', vertex_positions),
                                                        tex_coords=('f', overlay_tex.tex_coords))
        
        foreground.draw()
        
        
        
        
        # Load the emotion vectors-------------------------------------------------------------------------------------------------------------------
        emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sad', 'Surprise']
        
        if emotion_vectors is not None:
            
            
            emotion_in, generated_emotion = emotion_vectors
            emotion_in = emotion_in[0].tolist()  # Assuming emotion_in is a tensor
            emotion_out = generated_emotion[0].tolist()  # Assuming generated_emotion is a tensor

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
        
    
        

# glitch effect

def create_fullscreen_quad():
    # Coordinates for a fullscreen quad (two triangles)
    return [-1, -1, 1, -1, 1, 1, -1, 1]



# Fragment Shader for Glitch Effect
_glitch_fragment_shader_source ="""
#version 430 core
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(rgba32f) uniform image2D img_output;
uniform float time;
uniform vec2 resolution;

// Function to generate random values
float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    ivec2 texel_coord = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(texel_coord) / resolution.xy;

    // Glitch effect parameters
    float glitchIntensity = abs(sin(time)); // Varies between 0 and 1 over time
    float stripeWidth = 0.02; // Width of the glitch stripes
    float noiseIntensity = (rand(uv + time) * glitchIntensity)/2; // Random noise intensity

    // Calculate stripe pattern based on the y-coordinate
    float stripes = step(0.5 + glitchIntensity * 0.5, fract(uv.y / stripeWidth));
    
    // Base color variation for glitch
    vec3 colorShift = vec3(rand(uv + time), rand(uv - time), rand(uv * time));
    
    vec3 color = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), stripes); 
    
    // Apply stripes and noise to the color
    color = vec3(1.0, 1.0, 1.0) * stripes * noiseIntensity + colorShift * (1.0 - stripes);

    // Output the final color with some transparency
    imageStore(img_output, texel_coord, vec4(color, 0.5));
}

"""

if __name__ == '__main__':
    import queue

    # Read data from a JSON file
    with open('data/data.json', 'r') as file:
        loaded_data = json.load(file)

    # Accessing the data
    unnorm_out = loaded_data["unnorm_out"]
    max_x = loaded_data["max_x"]
    max_y = loaded_data["max_y"]
    
    window = pyglet.window.Window(int(max_x) + 50, int(max_y) + 50)
    frame_queue = queue.Queue()  # Queue to hold individual frames
    
    # Preload frames into the queue
    for frame in unnorm_out:
        frame_queue.put(frame)

    def update(dt):
        global frame_index
        if not frame_queue.empty():
            frame_data = frame_queue.get()  # Get the next frame from the queue
            visualise_body(frame_data, None, max_x, max_y, window, start_time, frame_index)  # Visualize it
            frame_index += 1
        else:
            pyglet.app.exit()
    
    pyglet.clock.schedule_interval(update, 0.15)  # Adjust interval as needed
    
    # Required in global scope ----------------------------------------
    
    start_time = time.time()

    global frame_index
    frame_index = 0
    
    start_time = time.time()
    
    # Required in global scope ----------------------------------------

    pyglet.app.run()