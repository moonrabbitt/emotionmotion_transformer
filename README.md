# Audience Interactive Emotion-Motion Multimodal Transformer

## Project Overview
This repository contains the EmotionMotion Interactive Transformer. This innovative system integrates real-time emotional feedback from YouTube live chat into interactive performance art. The art is then rebroadcast on the same YouTube live stream, creating a closed loop with OBS Studio. Leveraging affective computing datasets, the system processes chat sentiment to shape the motion and narrative of the resulting artwork.

## Repository Structure

## Motion Data

## Documentation Log

Over three months, the final product evolved through ten primary prototyping stages. The documentation log is organized according to these stages, further subdivided into weekly updates. Below are links to detailed documentation for each stage, providing a clear tracking of the project's development.

Initially, the project aimed to create an emotional dance sequencer model. However, due to the scarcity of emotionally labeled dance datasets, it pivoted towards affective computing gestures.

[**Prototype 1**](documentation/prototype_1.md)

Prototype 1 served as a proof of concept, focusing on establishing the broadcast loop.

[**Prototype 2**](documentation/prototype_2.md)

This stage involved testing various architectures, including LSTM and GANs.

[**Prototype 3**](documentation/prototype_3.md)

The third prototype introduced the Transformer-Sequencer. This component, built from scratch based on Andrej Karpathy's transformer tutorial, was adapted for non-categorical data like keypoints. The stage included extensive data preprocessing and transformer enhancements, such as normalization layers and dropout layers to prevent overfitting.

[**Prototype 4**](documentation/prototype_4.md)

An attempt was made to shift the model from absolute coordinates to delta coordinates between T and T+1 frames. Unfortunately, this approach faced challenges, possibly due to normalization issues.

[**Prototype 5**](documentation/prototype_5.md)

With the transformer reaching a satisfactory level, it was integrated into the live loop with YouTube chat. However, the generated motion became stagnant and repetitive, posing a significant challenge.

[**Prototype 6**](documentation/prototype_6.md)

Prototype 6 aimed to resolve the model's tendency to get stuck by penalizing small movements and applying L1, L2 normalizations.

[**Prototype 7**](documentation/prototype_7.md)

This prototype transformed the transformer model from single modality to multimodal. The initial design required emotions to be appended to each frame, processed alongside the sequencer. This was later altered to process emotions separately from frames, merging them only at the sequencer's end.

[**Prototype 8**](documentation/prototype_8.md)

Prototype 8 introduced a Mixture Density Network (MDN) into the Transformer-Sequencer, allowing the model to output a range of potential movements rather than a singular output. This addition was intended to prevent the model from getting stuck and to enhance control over the output.

[**Prototype 9**](documentation/prototype_9.md)

The ninth prototype explored the balancing of MDN loss and MSE loss to maximize the advantages of both. It also included additional data exploration analysis.
