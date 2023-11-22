import pyglet
import math
import time
from pyglet.shapes import Circle
import json

def visualise_body(all_frames, max_x, max_y, max_frames=500):
    # Pyglet window initialization
    window = pyglet.window.Window(int(max_x) + 50, int(max_y) + 50)

    # Define the limb connections based on keypoints
    # limb_connections = [
    #     (0, 1), (1, 2), (2, 3), (3, 4),
    #     (1, 5), (5, 6), (6, 7),
    #     (1, 8), (8, 9), (9, 10), (10, 11),
    #     (8, 12), (12, 13), (13, 14),
    #     (0, 15), (15, 16), (0, 17), (17, 18),
    #     (14, 19), (14, 20), (14, 21),
    #     (11, 22), (11, 23), (11, 24)
    # ]
    
    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                    'L-Elb', 'L-Wr', 'MidHip', 'R-Hip', 'R-Knee', 'R-Ank', 
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 
                    'L-Ear', 'L-BigToe', 'L-SmallToe', 'L-Heel', 'R-BigToe', 
                    'R-SmallToe', 'R-Heel']

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
        ("R-Ank", "R-Heel")
    ]

    # Convert limb connection names to index pairs
    limb_connections = [(keypointsMapping.index(start), keypointsMapping.index(end)) for start, end in limb_connections_names]

    # Load the same image for all limb connections
    limb_image_path = 'data/leg.png'
    limb_image = pyglet.image.load(limb_image_path)
    # Set anchor points
    limb_image.anchor_x = limb_image.width // 2  # Center of the leg image
    limb_image.anchor_y = 0  # Top of the leg image
    limb_sprites = {limb: pyglet.sprite.Sprite(limb_image) for limb in limb_connections}

    # Function to draw each frame
    def draw_frame(frame_data):
        
        for i in range(25):  # Assuming 25 keypoints
            x = frame_data[i * 2]
            y = max_y - frame_data[i * 2 + 1]
            circle = Circle(x, y, 5, color=(0, 255, 0))
            circle.draw()
        
        
        
        for limb, sprite in limb_sprites.items():
            start_idx, end_idx = limb
            start_x, start_y = frame_data[start_idx * 2], max_y - frame_data[start_idx * 2 + 1]
            end_x, end_y = frame_data[end_idx * 2], max_y - frame_data[end_idx * 2 + 1]  # y needs to invert
            
            # Reset sprite properties

            
            sprite.x = 0
            sprite.y = 0
            sprite.rotation = 0
            sprite.scale = 1


            # # Calculate midpoint, angle, and distance
            # mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
            # angle = -(math.atan2(start_y - end_y, end_x - start_x) + math.pi / 2)  # Flipped y-coordinates means we also change the order here
            angle = (math.atan2(start_y - end_y, end_x - start_x) +(math.pi/2))  # Flipped y-coordinates means we also change the order here
            distance = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

            # Set sprite properties
            
            sprite.rotation = math.degrees(angle)
            
            sprite.scale = distance / sprite.height
            
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
        nonlocal frame_index
        frame_index += 1
        if frame_index >= len(all_frames):
            pyglet.app.exit()

    # Schedule update
    pyglet.clock.schedule_interval(update, 0.25)

    # Run the Pyglet application
    frame_index = 0
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
    # Access other variables in a similar way


    # Example usage
    all_frames = unnorm_out # Your frames data


    visualise_body(all_frames, max_x, max_y)
