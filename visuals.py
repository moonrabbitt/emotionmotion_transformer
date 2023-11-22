import pyglet
import math

# Close any open windows (if any exist)
for window in pyglet.app.windows:
    window.close()

# Create the window first
window = pyglet.window.Window(800, 600)  # Set the window size to 800x600

# Set the window background color to black
pyglet.gl.glClearColor(0,0,0,0)

# Load the image
image = pyglet.image.load('data/leg.png')
sprite = pyglet.sprite.Sprite(image)

# Set the position and scale of the image
coord1 = (200, 500)  # Replace with your first coordinate
coord2 = (200, 100)  # Replace with your second coordinate

sprite.x = min(coord1[0], coord2[0])
sprite.y = min(coord1[1], coord2[1])

# Calculate the angle and set the rotation of the sprite
angle = math.atan2(coord2[1] - coord1[1], coord2[0] - coord1[0])
sprite.rotation = -math.degrees(angle)

# Calculate the distance between the two points and set the scale of the sprite
distance = math.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
sprite.scale = distance / max(sprite.width, sprite.height)

@window.event
def on_draw():
    window.clear()
    sprite.draw()

if __name__ == '__main__':
    pyglet.app.run()    