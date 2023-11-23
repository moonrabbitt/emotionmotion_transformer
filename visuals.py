import pyglet
import math
import time
from pyglet.shapes import Circle
import json

# Constants for blinking
BLINK_INTERVAL = 10  # The number of frames before a blink happens
BLINK_DURATION = 2   # The number of frames the blink lasts


def visualise_body(all_frames, max_x, max_y, max_frames=500):
    global frame_index
    frame_index = 0

    # Pyglet window initialization
    window = pyglet.window.Window(int(max_x) + 50, int(max_y) + 50)
    
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
        
        print(is_blinking)
        print(frame_index)
        print(frame_index % BLINK_INTERVAL)
        
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
                
                elif limb == (25,) and not is_blinking: # L-Pupil
                    sprite.width = eye_width
                    sprite.height = sprite.width
                    sprite.scale = 0.01
                    limb = (15,)
                
                elif limb == (26,) and not is_blinking: # R-Pupil
                    sprite.width = eye_width
                    sprite.height = sprite.width
                    sprite.scale = 0.01
                    limb = (16,)
                  
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
        if frame_index < len(all_frames):
            draw_frame(all_frames[frame_index])

    # Update function for animation
    def update(dt):
        global frame_index
        frame_index += 1
        if frame_index >= len(all_frames):
            pyglet.app.exit()

    # Schedule update
    pyglet.clock.schedule_interval(update, 0.25)

    # Run the Pyglet application
    pyglet.app.run()


if __name__ == '__main__':
    
    
    
    # Read data from a JSON file
    with open('data/data.json', 'r') as file:
        loaded_data = json.load(file)

    # Accessing the data
    unnorm_out = loaded_data["unnorm_out"]
    min_x = loaded_data["min_x"]
    max_x = loaded_data["max_x"]
    min_y = loaded_data["min_y"]
    max_y = loaded_data["max_y"]

    # Example usage
    all_frames = unnorm_out # Your frames data


    visualise_body(all_frames, max_x, max_y)
