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


def select_shader(emotion):
    if emotion == 'Sad':
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

    elif emotion == 'Happiness':
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

    elif emotion == 'Surprise':
        _compute_source = """#version 430 core
        // adapted from https://github.com/genekogan/Processing-Shader-Examples/blob/master/ColorShaders/data/rain.glsl
        layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        layout(rgba32f) uniform image2D img_output;
        uniform float time;
        uniform float thingh;
        //uniform float fade;
        uniform float slow;
        uniform float thinggris;
        uniform vec2 thingres;


        // Noise generation functions

        vec4 mod289(vec4 x) {
            return x - floor(x * (1.0 / 289.0)) * 289.0;
        }

        vec4 permute(vec4 x) {
            return mod289(((x * 34.0) + 1.0) * x);
        }

        vec4 taylorInvSqrt(vec4 r) {
            return 1.79284291400159 - 0.85373472095314 * r;
        }

        vec2 fadeEffect(vec2 t) {
            return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        }

        float cnoise(vec2 P) {
            vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
            vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
            Pi = mod289(Pi); // To avoid truncation effects in permutation
            vec4 ix = Pi.xzxz;
            vec4 iy = Pi.yyww;
            vec4 fx = Pf.xzxz;
            vec4 fy = Pf.yyww;

            vec4 i = permute(permute(ix) + iy);

            vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
            vec4 gy = abs(gx) - 0.5 ;
            vec4 tx = floor(gx + 0.5);
            gx = gx - tx;

            vec2 g00 = vec2(gx.x,gy.x);
            vec2 g10 = vec2(gx.y,gy.y);
            vec2 g01 = vec2(gx.z,gy.z);
            vec2 g11 = vec2(gx.w,gy.w);

            vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
            g00 *= norm.x;
            g01 *= norm.y;
            g10 *= norm.z;
            g11 *= norm.w;

            float n00 = dot(g00, vec2(fx.x, fy.x));
            float n10 = dot(g10, vec2(fx.y, fy.y));
            float n01 = dot(g01, vec2(fx.z, fy.z));
            float n11 = dot(g11, vec2(fx.w, fy.w));

            vec2 fade_xy = fadeEffect(Pf.xy);
            vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
            float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
            return 2.3 * n_xy;
        }

        // Classic Perlin noise, periodic variant
        float pnoise(vec2 P, vec2 rep)
        {
        vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
        vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
        Pi = mod(Pi, rep.xyxy); // To create noise with explicit period
        Pi = mod289(Pi); // To avoid truncation effects in permutation
        vec4 ix = Pi.xzxz;
        vec4 iy = Pi.yyww;
        vec4 fx = Pf.xzxz;
        vec4 fy = Pf.yyww;

        //  vec4 i = permute(permute(ix) + iy);
        vec4 i = permute(permute(iy + ix) + ix + iy + permute(ix));

        vec4 gx = fract(i * (1.0 / (20.0+31.0*sin(0.1*time)))) * 2.0 - 1.0 ;
        vec4 gy = abs(gx) - 0.5 * sin(gx*time*0.1) ;
        vec4 tx = floor(gx + 0.5);
        gx = gx - tx;

        vec2 g00 = vec2(gx.x,gy.x);
        vec2 g10 = vec2(gx.y,gy.y);
        vec2 g01 = vec2(gx.z,gy.z);
        vec2 g11 = vec2(gx.w,gy.w);

        vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
        g00 *= norm.x;
        g01 *= norm.y;
        g10 *= norm.z;
        g11 *= norm.w;

        float n00 = dot(g00, vec2(fx.x, fy.x));
        float n10 = dot(g10, vec2(fx.y, fy.y));
        float n01 = dot(g01, vec2(fx.z, fy.z));
        float n11 = dot(g11, vec2(fx.w, fy.w));

        vec2 fade_xy = fadeEffect(Pf.xy);
        vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
        float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
        return 1.3 * n_xy;
        }


        void main() {
            ivec2 texel_coord = ivec2(gl_GlobalInvocationID.xy);

            float time2 = (sin(time)+1)*10.0/slow;  // Oscillates between 0 and 1
            vec3 rgb = vec3(vec2(float(texel_coord.x)/thingres.x,float(texel_coord.y)/thingres.y), 0.1);
            rgb.g *= time2 * pnoise(vec2(rgb.rg + time2), vec2(rgb.rb));
            rgb.r = sin(time2 * thingh) + rgb.r;
            rgb.b = (cos(thingh+0.1) * time2) + rgb.b;
            vec3 col = mix(rgb, vec3(0.33333 * (rgb.r + rgb.b + rgb.g)), (thinggris));
            //apparently it likes thing as uniform names. why is this? I don't know. WHAT?
            //col = vec3(0.0,1.0,1.0);

            vec4 output_color = vec4(col,1.0);


            imageStore(img_output, texel_coord, output_color);
        }

        """

    elif emotion == 'Fear':
        _compute_source = """ 
        #version 430 core
        //adapted from https://glslsandbox.com/e#107623.0
        layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        layout(rgba32f) uniform image2D img_output;

        uniform float time;
        uniform vec2 resolution;
        uniform vec2 position;

        #define MOD3 vec3(443.8975, 397.2973, 491.1871) // uv range
        #define PI 3.14159265

        #define res resolution.xy


        #define MOD3 vec3(443.8975, 397.2973, 491.1871) // uv range
        float hash11(float p) {
            vec3 p3  = fract(vec3(p) * MOD3);
            p3 += dot(p3, p3.yzx + 19.191);
            return fract((p3.x + p3.y) * p3.z);
        }
        float sHash11(float a) {
            return
                mix(
                    hash11(floor(a)),
                    hash11(floor(a)+1.),
                    smoothstep(0., 1., fract(a))
                );
        }

        //	Classic Perlin 2D Noise 
        //	by Stefan Gustavson
        //
        //  repetition of x added - stb
        vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
        vec2 fade(vec2 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}

        float cnoise(vec2 P, float rep){
            P.x = mod(P.x, rep); // x rep 1/2

        vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);

            Pi.z = mod(Pi.z, rep); // x rep 2/2

        vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
        Pi = mod(Pi, 289.0); // To avoid truncation effects in permutation
        vec4 ix = Pi.xzxz;
        vec4 iy = Pi.yyww;
        vec4 fx = Pf.xzxz;
        vec4 fy = Pf.yyww;
        vec4 i = permute(permute(ix) + iy);
        vec4 gx = 2.0 * fract(i * 0.0243902439) - 1.0; // 1/41 = 0.024...
        vec4 gy = abs(gx) - 0.5;
        vec4 tx = floor(gx + 0.5);
        gx = gx - tx;
        vec2 g00 = vec2(gx.x,gy.x);
        vec2 g10 = vec2(gx.y,gy.y);
        vec2 g01 = vec2(gx.z,gy.z);
        vec2 g11 = vec2(gx.w,gy.w);
        vec4 norm = 1.79284291400159 - 0.85373472095314 * 
            vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
        g00 *= norm.x;
        g01 *= norm.y;
        g10 *= norm.z;
        g11 *= norm.w;
        float n00 = dot(g00, vec2(fx.x, fy.x));
        float n10 = dot(g10, vec2(fx.y, fy.y));
        float n01 = dot(g01, vec2(fx.z, fy.z));
        float n11 = dot(g11, vec2(fx.w, fy.w));
        vec2 fade_xy = fade(Pf.xy);
        vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
        float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
        return 2.3 * n_xy;
        }

        float Eye(vec2 p, float pupil, vec2 lpos) {
            pupil += .1*(1.-length(lpos))-.1;


            //lpos.x = sin(time);

            // radial coords
            vec2 pr = vec2(atan(p.x, p.y) / PI / 2., clamp((length(p)-1.)/pupil+.8, 0., 1.));

            // smooth curve from pupil to outer iris
            pr.y = smoothstep(0., 1., pr.y);

            // noise frequency for radial coords
            vec2 freq = vec2(30., 1.5);

            // radial noise
            float f = pow((cnoise(pr*freq, freq.x)+1.)/4., .5);

            // more radial noise
            f -= 1.*pow((cnoise(pr*freq*vec2(2., 3.)+9., 2.*freq.x)+1.)/2.-.5, 2.);

            //vec2 lpos = vec2(.5, .75);

            // general shading
            float shade = dot(p, lpos);

            // lightening of iris
            f -= .7 * shade;

            // darker inner iris & pupil
            f *= pow(smoothstep(0., .5, pr.y), .15);

            // darker ring around iris
            f = mix(f, .25, smoothstep(0.5, 1., pr.y+.2));

            // mix in sclera
            f = mix(f, 1.-.2*dot(p, p)+.75*shade, smoothstep(0.7, .85, pr.y));

            // highlight
            f = mix(1., f, clamp((length(p-lpos/1.)-.15)/.025, 0., 1.));

            // eyelids
           f = mix(f, 0., clamp((length(vec2(p.x, abs(p.y))+vec2(0., 1.3))-2.15)/.04, 0., 1.));

            return f;
        }

        void main( void ) {
            ivec2 texel_coord = ivec2(gl_GlobalInvocationID.xy);

            vec2 p = 10.* (vec2(texel_coord) - resolution / 2.0) / resolution.y;

            // pupil contraction/expansion
            float t = 1.+.05*sHash11(1.5*time);

            vec2 position2 = vec2(sin(position.x),sin(position.y));
            vec4 value = vec4( vec3(Eye(p, t, position2-.5)), 1.0 );

        imageStore(img_output, texel_coord, value);

        }


        """

    elif emotion == 'Anger':
        _compute_source = """#version 430 core

        layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        layout(rgba32f) uniform image2D img_output; // Output image

        uniform vec2 resolution; // Texture resolution

        void main() {
            ivec2 texel_coord = ivec2(gl_GlobalInvocationID.xy); // Get the coordinate of the current texel
            if (texel_coord.x >= resolution.x || texel_coord.y >= resolution.y) return; // Check boundary

            // Define two colors for interpolation
            vec3 colorA = vec3(1.0, 0.0, 0.0); // Red
            vec3 colorB = vec3(0.0, 0.0, 1.0); // Blue

            // Calculate the interpolation factor
            float factor = smoothstep(0.0, resolution.x, float(texel_coord.x));

            // Interpolate between colorA and colorB
            vec3 color = mix(colorA, colorB, factor);

            // Store the computed color in the image
            imageStore(img_output, texel_coord, vec4(color, 1.0));
        }

"""
    else:
        _compute_source = """#version 430 core
        layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        layout(rgba32f) uniform image2D img_output;
        uniform float time;
        uniform float slow;



        vec4 mod289(vec4 x) {
            return x - floor(x * (1.0 / 289.0)) * 289.0;
        }

        vec4 permute(vec4 x) {
            return mod289(((x * 34.0) + 1.0) * x);
        }

        vec4 taylorInvSqrt(vec4 r) {
            return 1.79284291400159 - 0.85373472095314 * r;
        }

        vec2 fadeEffect(vec2 t) {
            return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        }

        float cnoise(vec2 P, float rep) {
            P.x = mod(P.x, rep);

            vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);

            Pi.z = mod(Pi.z, rep);
            vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
            Pi = mod289(Pi); // To avoid truncation effects in permutation
            vec4 ix = Pi.xzxz;
            vec4 iy = Pi.yyww;
            vec4 fx = Pf.xzxz;
            vec4 fy = Pf.yyww;

            vec4 i = permute(permute(ix) + iy);

            vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0 ;
            vec4 gy = abs(gx) - 0.5 ;
            vec4 tx = floor(gx + 0.5);
            gx = gx - tx;

            vec2 g00 = vec2(gx.x,gy.x);
            vec2 g10 = vec2(gx.y,gy.y);
            vec2 g01 = vec2(gx.z,gy.z);
            vec2 g11 = vec2(gx.w,gy.w);

            vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
            g00 *= norm.x;
            g01 *= norm.y;
            g10 *= norm.z;
            g11 *= norm.w;

            float n00 = dot(g00, vec2(fx.x, fy.x));
            float n10 = dot(g10, vec2(fx.y, fy.y));
            float n01 = dot(g01, vec2(fx.z, fy.z));
            float n11 = dot(g11, vec2(fx.w, fy.w));

            vec2 fade_xy = fadeEffect(Pf.xy);
            vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
            float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
            return 2.3 * n_xy;
        }

        // Classic Perlin noise, periodic variant
        float pnoise(vec2 P, vec2 rep)
        {
        vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
        vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
        Pi = mod(Pi, rep.xyxy); // To create noise with explicit period
        Pi = mod289(Pi); // To avoid truncation effects in permutation
        vec4 ix = Pi.xzxz;
        vec4 iy = Pi.yyww;
        vec4 fx = Pf.xzxz;
        vec4 fy = Pf.yyww;

        //  vec4 i = permute(permute(ix) + iy);
        vec4 i = permute(permute(iy + ix) + ix + iy + permute(ix));

        vec4 gx = fract(i * (1.0 / (20.0+31.0*sin(0.1*time)))) * 2.0 - 1.0 ;
        vec4 gy = abs(gx) - 0.5 * sin(gx*time*0.1) ;
        vec4 tx = floor(gx + 0.5);
        gx = gx - tx;

        vec2 g00 = vec2(gx.x,gy.x);
        vec2 g10 = vec2(gx.y,gy.y);
        vec2 g01 = vec2(gx.z,gy.z);
        vec2 g11 = vec2(gx.w,gy.w);

        vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
        g00 *= norm.x;
        g01 *= norm.y;
        g10 *= norm.z;
        g11 *= norm.w;

        float n00 = dot(g00, vec2(fx.x, fy.x));
        float n10 = dot(g10, vec2(fx.y, fy.y));
        float n01 = dot(g01, vec2(fx.z, fy.z));
        float n11 = dot(g11, vec2(fx.w, fy.w));

        vec2 fade_xy = fadeEffect(Pf.xy);
        vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
        float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
        return 1.3 * n_xy;
        }



        void main() {
        ivec2 texel_coord = ivec2(gl_GlobalInvocationID.xy);

        // Base color change on position
        float base_red = float(texel_coord.x) / (gl_NumWorkGroups.x);
        float base_green = float(texel_coord.y) / (gl_NumWorkGroups.y);

        // Modulate color based on time
        float time_red = (sin(time) + 1.0) ;  // Oscillates between 0 and 1
        float time_green = (cos(time) + 1.0) ;  // Oscillates between 0 and 1

        float noise = pnoise(vec2(time_red,time_green),vec2(base_red,base_green));
        float noise1 = pnoise(vec2(time_red,time_green),vec2(time_green,time_red));

        noise = cnoise(vec2(noise1,noise));      
        vec2 rg = vec2(time_red,time_green)*noise + vec2(time_red,time_green)*noise1 ;

        // Combine the position-based color with the time-based modulation
        //vec4 value = vec4(base_red * time_red, base_green * time_green, 0.0, 1.0);
        vec4 value = vec4(rg * vec2(time_red,time_green), 0.0, 1.0);

        imageStore(img_output, texel_coord, value);
        }
        """

    return _compute_source


def set_uniforms_for_shader(emotion, shader_program,args):
    if emotion == 'Sad':
        start_time = args
        shader_program['time'] = (time.time() - start_time)

    elif emotion == 'Happiness':
        start_time = args
        current_time = (time.time() - start_time) * 0.6
        shader_program['time'] = current_time

    elif emotion == 'Surprise':
        start_time = args
        shader_program['time'] = float(time.time() - start_time)
        shader_program['thingh'] = 1.0
        # shader_program['fade'] = 1.0
        shader_program['slow'] = 10.0

        # print(shader_program['colour_h'])
        shader_program['thinggris'] = 1.0
        shader_program['thingres'] = (float(window.width), float(window.height))
        # print(shader_program['colour_h'])

    elif emotion == 'Fear':
        start_time = args
        shader_program['time'] = float(time.time() - start_time)
        shader_program['resolution'] = (float(window.width), float(window.height * 1.4))
        # shader_program['resolution'] = (2500.0,2500.0)
        shader_program['position'] = (window._mouse_x, window._mouse_y)
        print(shader_program['position'])

    elif emotion == 'Anger':
        shader_program['resolution'] = (float(window.width), float(window.height))

    else:
        start_time = args
        shader_program['time'] = float(time.time() - start_time)


def create_program(emotion):
    # Creating shaders and program
    vert_shader = Shader(_vertex_source, 'vertex')
    frag_shader = Shader(_fragment_source, 'fragment')
    shader_program = ShaderProgram(vert_shader, frag_shader)
    _compute_source = select_shader(emotion)
    compute_program = pyglet.graphics.shader.ComputeShaderProgram(_compute_source)

    return shader_program, compute_program


def shader_on_draw(emotion,shader_program, compute_program, batch,window,args):
    tex = pyglet.image.Texture.create(window.width, window.height, internalformat=GL_RGBA32F)
    tex.bind_image_texture(unit=compute_program.uniforms['img_output'].location)

    set_uniforms_for_shader(emotion, compute_program,args)

    with compute_program:
        compute_program.dispatch(tex.width, tex.height, 1, barrier=GL_ALL_BARRIER_BITS)

    group = RenderGroup(tex, shader_program)
    indices = (0, 1, 2, 0, 2, 3)
    vertex_positions = create_quad(0, 0, tex)

    vertex_list = shader_program.vertex_list_indexed(4, GL_TRIANGLES, indices, batch, group,
                                                     position=('f', vertex_positions),
                                                     tex_coords=('f', tex.tex_coords))

import time
if __name__ == '__main__':
    emotion = 'Fear'
    shader_program, compute_program = create_program(emotion)
    # Pyglet window setup

    window = pyglet.window.Window(width=1200, height=1800)
    # Main loop
    global start_time
    start_time = time.time()


    @window.event
    def on_draw():
        window.clear()
        batch = pyglet.graphics.Batch()

        shader_on_draw(emotion,shader_program, compute_program, batch,window,start_time)

        batch.draw()




    pyglet.app.run()