# Prototype 2
## Motion Recognition and Testing Different Architectures

### Week 3

Having established the basic proof of concept with the broadcast loop, my attention turned to developing the motion sequencer model. The first task was to acquire suitable data for model testing. I explored several vision recognition models to identify the best fit for capturing motion data.

My initial exploration led me to the [Holistic MediaPipe model](https://developers.google.com/mediapipe/solutions/vision/holistic_landmarker). However, I found that it was limited to recognizing only one person at a time, which was not ideal for my project's needs. I then considered [YOLOv8](https://docs.ultralytics.com), which proved to be fast and effective for live multi-person scenarios, but it only offered 14 keypoints and crucially lacked foot keypoint recognition, a significant drawback for dance motion capture.

Consequently, I shifted to using the [OpenPose BODY 25 model](https://docs.ultralytics.com), which includes feet keypoint recognition. Implementing OpenPose proved challenging, primarily due to its cmake backend, which necessitated additional installation steps. Despite these difficulties, its comprehensive keypoint detection made it the most suitable choice for the motion sequencer model.


### Week 4 & 5

After choosing the OpenPose model for keypoint detection, my next objective was to test various architectures for the motion sequencer. I began with a straightforward LSTM (Long Short-Term Memory) model. This model showed promise in recognizing motion, yet the output exhibited some noise and lacked smoothness. The code for this LSTM implementation is available [here](notebooks/prototypes/basic-prototype-2-LSTM.ipynb), and you can view the results [here](FILL IN LINK).

In pursuit of refinement, I ventured into integrating a Generative Adversarial Network (GAN) with the model. The code for this experiment is located [here](notebooks/prototypes/basic-prototype-2-GAN.ipynb). This endeavor was inspired by [Saito et al. (2017)](http://arxiv.org/abs/1811.09245) and their work on temporal GANs capable of generating videos. However, this integration was fraught with challenges. The generator produced overwhelmingly high outputs, while the discriminator was markedly ineffective, as shown [here](PIC HERE). Further research revealed that GANs are not typically preferred for sequence modeling. Their complexity in creating coherent sequences and the difficulty in training them for temporally consistent outputs are significant hurdles. Notably, successful applications of GANs in motion prediction, such as those by [Zhao et al (2023)](https://dl.acm.org/doi/10.1145/3579359) or [Xu et al. (2022)](http://arxiv.org/abs/2203.07706) often pair them with transformers or LSTM models to maintain temporal coherence.

After dedicating over a week to this approach with limited success, I decided it was time to reevaluate and consider other architectural options that could provide a more reliable and efficient solution for the motion sequencer.

