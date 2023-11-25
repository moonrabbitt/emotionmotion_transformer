import pyglet
import time
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.gl import *
from pyglet.graphics import Group

class RenderGroup(Group):
    def __init__(self, texture, program, order=0, parent=None):
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

def create_quad(x, y, texture):
    x2 = x + texture.width
    y2 = y + texture.height
    return x, y, x2, y, x2, y2, x, y2

# Vertex Shader
_vertex_source = """
#version 330 core
in vec2 position;
in vec3 tex_coords;
out vec3 texture_coords;

uniform WindowBlock
{
    mat4 projection;
    mat4 view;
} window;

void main()
{
    gl_Position = window.projection * window.view * vec4(position, 1, 1);
    texture_coords = tex_coords;
}
"""

# Fragment Shader
_fragment_source = """
#version 330 core
in vec3 texture_coords;
out vec4 final_colors;

uniform sampler2D our_texture;

void main()
{
    final_colors = texture(our_texture, texture_coords.xy);
}
"""

# Compute Shader
_compute_source = """
#version 430 core
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(rgba32f) uniform image2D img_output;
uniform float time;

float random (in vec2 _st) {
    return fract(sin(dot(_st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float noise (in vec2 _st) {
    vec2 i = floor(_st);
    vec2 f = fract(_st);

    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
           (c - a)* u.y * (1.0 - u.x) +
           (d - b) * u.x * u.y;
}

#define NUM_OCTAVES 5

float fbm ( in vec2 _st) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    // Rotate to reduce axial bias
    mat2 rot = mat2(cos(0.5), sin(0.5),
                    -sin(0.5), cos(0.50));
    for (int i = 0; i < NUM_OCTAVES; ++i) {
        v += a * noise(_st);
        _st = rot * _st * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}



void main() {
    ivec2 texel_coord = ivec2(gl_GlobalInvocationID.xy);

    vec2 st = vec2(texel_coord) / imageSize(img_output).xy;

    vec2 q;
    q.x = fbm(st + 0.00 * time);
    q.y = fbm(st + vec2(1.0));
    
    vec2 p1 = vec2(fbm(st + q + vec2(1.7,9.2) + 0.15 * time));
    vec2 p2 = vec2(fbm(st + q + vec2(8.3,2.8) + 0.126 * time));

    vec2 r;
    r.x = fbm(p1 + fbm(p1 + fbm(p1)));
    r.y = fbm(p2 + fbm(p2 + fbm(p2)));

    float f = fbm(st + r);
    vec3 baseColor = vec3(0.0, 0.0, 0.0); // Black
    vec3 color = mix(vec3(0.101961,0.619608,0.666667), // Black
                vec3(0.666667,0.666667,0.498039),
                clamp((f*f)*4.0,0.0,1.0));
                
    color = mix(color,
                vec3(0,0,0.164706),
                clamp(length(q),0.0,1.0));
                
    color = mix(color,
                vec3(0.666667,1,1),
                clamp(length(r.x),0.0,1.0));
    
    color = (f*f*f + .6*f*f + .5*f) * color;

    imageStore(img_output, texel_coord, vec4(color, 1.0));
}

"""

# Creating shaders and program
vert_shader = Shader(_vertex_source, 'vertex')
frag_shader = Shader(_fragment_source, 'fragment')
shader_program = ShaderProgram(vert_shader, frag_shader)
compute_program = pyglet.graphics.shader.ComputeShaderProgram(_compute_source)

# Pyglet window setup
window = pyglet.window.Window(width=1200, height=1800)

@window.event
def on_draw():
    window.clear()
    batch = pyglet.graphics.Batch()

    tex = pyglet.image.Texture.create(window.width, window.height, internalformat=GL_RGBA32F)
    tex.bind_image_texture(unit=compute_program.uniforms['img_output'].location)
    
    current_time = (time.time() - start_time) 
    compute_program['time'] = float(current_time)
    # compute_program['resolution'] = (float(window.width),float( window.height))
    # compute_program['mouse'] = (0, 0)
    
    with compute_program:
        compute_program.dispatch(tex.width, tex.height, 1, barrier=GL_ALL_BARRIER_BITS)

    group = RenderGroup(tex, shader_program)
    indices = (0, 1, 2, 0, 2, 3)
    vertex_positions = create_quad(0, 0, tex)

    vertex_list = shader_program.vertex_list_indexed(4, GL_TRIANGLES, indices, batch, group,
                                                     position=('f', vertex_positions),
                                                     tex_coords=('f', tex.tex_coords))
    batch.draw()

# Main loop
start_time = time.time()
pyglet.app.run()
