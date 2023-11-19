# Audience Interactive Emotion-Motion Multimodal Transformer


## Project Overview
This repository hosts the EmotionMotion Interactive Transformer, a system designed to integrate live audience emotional feedback from Youtube live chat into interactive performance art, which is broadcasted back into the same YouTube live as a closed loop using OBS studios. Utilizing affective computing datasets, the Sequencer processes real-time chat sentiment to influence the generated motion and narrative of the outputted artwork.



## Repository Structure

## Motion Data





## Documentation Log

The final product went through 10 main stages of prototyping over the course of 3 months. The documentation log is therefore broken down into these prototype stages, which is further broken down into a weekly blog format. The links to the documentations can be found below, they are broken down into individual files with each file pertaining to each stage of prototyping for ease of tracking.

[<u>Prototype 1</u>](documentation/prototype_1.md)

Prototype 1 is the proof of concept and focuses on getting the broadcast loop working.

[<u>Prototype 2</u>](documentation/prototype_2.md)

Prototype 2 is the first iteration of the Transformer-Sequencer, which is the main component of the project. The transformer was coded from scratch based on Andrej Karpathy's tutorial on transformers but adapted to be used with keypoints which is non-categorical. 

[<u>Prototype 3</u>](documentation/prototype_3.md)

Added a lot of data preprocessing and improved transformer through normalisation layers and dropout layers to prevent overfitting.


[<u>Prototype 4</u>](documentation/prototype_4.md)

Tried to move model from absolute coordinates to delta between T and T+1 frames to allow more control over the model. However, this did not work out, possibly due to normalisation issues.

(<u>Prototype 5</u>)[documentation/prototype_5.md]

Transformer got to a good point, so integrating with live loop with live youtube chat. However when integrating with live loop, the motion generated became very stagnant and repetitive. This presents a major problem.

(<u>Prototype 6</u>)[documentation/prototype_6.md]

Prototype 6 attempts fix model getting stuck by penalising small movements and through L1, L2 normalisations.

(<u>Prototype 7</u>)[documentation/prototype_7.md]

Prototype 7 converts transformer model from single modality to multimodal. Initially model required emotion to be concatenated to the end of every frame, and it is processed together throughout the entire sequencer. This was later changed to emotion being processed separately from the frames, and only concatenated at the end of the sequencer.

(<u>Prototype 8</u>)[documentation/prototype_8.md]

Prototype 8 adds MDN to the Transformer-Sequencer, which allows for the model to output a distribution of possible movements instead of a single movement. Hopefully this reduces the chances of the model being stuck and allows for more control.

(<u>Prototype 9</u>)[documentation/prototype_9.md]

Exploring scheduling MDN loss and MSE loss to maximise the benefits of both. Also added some data exploration analysis.

