# Prototype 5: Adding Delta and Refactoring

Access the full code for Prototype 5 [here](notebooks/prototypes/prototype-5.ipynb).

## Week 10 & 11

### Implementing Delta for Dynamic Motion

This period was dedicated to refining the previous prototype and adding a delta dimension (T+1 - T), which offers a more dynamic representation of motion than absolute coordinates. The integration of delta, representing the difference between consecutive frames, was aimed at enabling the model to learn the nuances of motion change.

### Challenges with Delta Implementation

However, incorporating delta led to suboptimal model outputs, with keypoints displaying an unnatural floating behavior. This issue, likely arising from a normalisation accumulation error, can be seen. Despite efforts to rectify this in the delta model, improvements were limited.

### Focus on Absolute Coordinate Outputs and Biological Plausibility

In response, I refocused on enhancing the absolute coordinate model, which was producing more emotionally relevant motion outputs. Additionally, I implemented sanity checks for biological plausibility in keypoints. This was particularly necessary due to implausible data points arising from extensive interpolation in the data preprocessing stage, such as instances where the nose would drop below the hip.


### Very good emotion-contexted movements when using validation data

The motion were very good, very clear contextually and very dynamic. However, this is only true when using validation data and I found that when I integrated the model with the live loop and the motion is not attached to the correct emotion (because the motion has to be the 3 frames of the last generated batch for it to look like one continous video), the model really gets stuck in transitional motion and just starts repeating itself. I will focus on this in the upcoming weeks.


### Transitional emotions

One interesting affective computing finding that I accidentally found when implementing the model to a live loop is the transitioning emotions. For example, happy motion labelled with a sad emotion lead to the model interpreting as anger. This reflects the latent space of affective computing motion and shows the limits of it HOW. You can see the results below:


### Visualisation Exploration with Stable Diffusion

Having achieved emotionally resonant motion outputs, I experimented with visualisation using stable diffusion models like [deforum](https://replicate.com/deforum/deforum_stable_diffusion) and [stablityAI](https://huggingface.co/spaces/stabilityai/stable-diffusion). The results, while not adhering perfectly to the input poses, yielded intriguing artworks, examples of this will be included in the thesis. I found that prompts resembling human figures adhered more effectively to the skeleton.

### Addressing Processing Speed and Future Plans

ControlNet's slow processing speed poses a challenge for live systems. Future visualisations may involve tools like Pygame or OpenCV, potentially with matrix augmentation. This approach aligns with the aesthetic of early video artists like Nam June Paik and Bill Viola. However, visualisation efforts will be temporarily paused to prioritise enhancing the model's dynamic motion capability.


### Integration with live loop

I integrated the results with live loop.
