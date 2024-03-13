# Audience Interactive Emotion-Motion Multimodal Transformer

## Project Overview
This repository contains the EmotionMotion Interactive Transformer+MDN multimodal hybrid model which is part of my thesis for MA/MSc Computing and the Creative Industries (Modular) for the University of Arts London. 

This innovative system integrates real-time emotional feedback from YouTube live chat into interactive performance art. The art is then rebroadcast on the same YouTube live stream, creating a closed loop with OBS Studio. Leveraging affective computing datasets, the system processes chat sentiment to shape the motion and narrative of the resulting artwork.

The architecture of the model is shown below: 
![model architecture](https://github.com/moonrabbitt/emotionmotion_transformer/blob/main/data/Dissertation%20-%20Figure%20X_%20Architecture%20Transformer%20and%20MDN%20(2).jpg)

You will need OBS to be able to stream it back to YouTube, but it is not necessary for the project to run.

A short video showing the project's process and final outcome can be seen here: https://www.youtube.com/watch?v=bC6bmK3HHys 


## Repository Structure

  - **main.py**: The core script that manages parallel processing, including chat scraping, live chat sentiment analysis, AI model motion generation, and visualisations.
  - **model.py**: Contains the code for training and testing the Transformer+MDN model. Use this to retrain the transformer. Training typically takes about 8 hours on an RTX 3060 for 200k epochs.
  - **data.py**: Handles data preprocessing, normalisation, validation checks, and loading data for both main and model scripts.
  - **glsl.py**: Contains GLSL code for visualisations.
  - **visuals.py**: Manages sprite visualisations using Pyglet and integrates GLSL code from glsl.py.
  - **analysis.ipynb**: Notebook with analysis figures for the thesis, including data explorations.
  - **libs/mdn.py**: Contains the MDN code and all sampling algorithms.
  - **documentation/**: Directory for all weekly weblogs documenting the project development.
  - **additional_inputs/**: Includes code for adding extra features when displaying the work in a gallery setting.
   - **notebooks/**: Contains archived notebooks use to test various code snippets during development.




## Motion Data

I used the Multiview emotional expressions dataset by shang et al. 2019 and the DanceDB dataset from the University of Cyprus to train this model. I did not include the datasets in this repository as it is too large. However, the model should be able to train on any Openpose Body_25 outputs in the correct data format.

## Documentation Log

Over three months, the final product evolved through ten primary prototyping stages. The documentation log is organised according to these stages, further subdivided into weekly updates. Below are links to detailed documentation for each stage, providing a clear tracking of the project's development.

Initially, the project aimed to create an emotional dance sequencer model. However, due to the scarcity of emotionally labeled dance datasets, it pivoted towards affective computing gestures.

[**Prototype 1**](documentation/prototype_1.md)

Prototype 1 served as a proof of concept, focusing on establishing the broadcast loop.

[**Prototype 2**](documentation/prototype_2.md)

This stage involved testing various architectures, including LSTM and GANs.

[**Prototype 3**](documentation/prototype_3.md)

The third prototype introduced the Transformer-Sequencer. This component, built from scratch based on Andrej Karpathy's transformer tutorial, was adapted for non-categorical data like keypoints. The stage included extensive data preprocessing and transformer enhancements, such as normalisation layers and dropout layers to prevent overfitting.

[**Prototype 4**](documentation/prototype_4.md)

An attempt was made to shift the model from absolute coordinates to delta coordinates between T and T+1 frames. Unfortunately, this approach faced challenges, possibly due to normalisation issues.

[**Prototype 5**](documentation/prototype_5.md)

With the transformer reaching a satisfactory level, it was integrated into the live loop with YouTube chat. However, the generated motion became stagnant and repetitive, posing a significant challenge.

[**Prototype 6**](documentation/prototype_6.md)

Prototype 6 aimed to resolve the model's tendency to get stuck by penalising small movements and applying L1, L2 normalisations.

[**Prototype 7**](documentation/prototype_7.md)

This prototype transformed the transformer model from single modality to multimodal. The initial design required emotions to be appended to each frame, processed alongside the sequencer. This was later altered to process emotions separately from frames, merging them only at the sequencer's end.

[**Prototype 8**](documentation/prototype_8.md)

Prototype 8 introduced a Mixture Density Network (MDN) into the Transformer-Sequencer, allowing the model to output a range of potential movements rather than a singular output. This addition was intended to prevent the model from getting stuck and to enhance control over the output.

[**Prototype 9**](documentation/prototype_9.md)

The ninth prototype explored the balancing of MDN loss and MSE loss to maximise the advantages of both. It also included additional data exploration analysis.

[**Prototype 10**](documentation/prototype_10.md)

The final prototype integrated sprite visuals and parallel processes for live deployment.
