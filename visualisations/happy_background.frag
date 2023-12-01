"""#version 330 core

out vec4 FragColor;

void main() {
    // Create a simple linear gradient
    float gradient = gl_FragCoord.y / float(800); // Assuming a window height of 800 pixels
    vec3 color = mix(vec3(1.0, 0.8, 0.6), vec3(0.6, 0.8, 1.0), gradient);
    
    FragColor = vec4(color, 1.0);
}"""