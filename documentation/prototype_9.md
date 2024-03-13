# Prototype 9

## Week 16

## Emotion Conditioning

With the model now displaying more dynamic movements and avoiding repetition in live loops, I still faced a challenge: the emotion conveyed by the generated motions was somewhat muted, not as pronounced as in scenarios without the Mixture Density Network (MDN). To address this, I made some adjustments in the architecture.

I shifted from inputting raw data (x), which included emotions appended at the end, to using logits in the emotion dense layer (emotion_fc2). This was based on the observation that the model could potentially shortcut its learning by overly relying on the explicit emotion data at the end of x, rather than learning to interpret emotion from the motion data itself. By using logits, the intent was to compel the model to discern and replicate the emotion inherent in the motion. To further emphasise the importance of accurately capturing emotion, I amplified the emotion loss weight in the model's training regimen.

To refine the emotion conditioning, I also implemented attention pooling. This involved first passing the key points logits matrix through a dropout layer to lessen the impact of minor noise, followed by attention pooling to reduce its dimensions while concentrating on emotionally relevant features. This technique, borrowing from advances in speech model processing, utilises self-attention pooling to capture complex, distributed relationships within the sequence, thereby enhancing the model's ability to generate motions that are emotionally resonant. The streamlined data then proceeds through several linear layers with ReLU activation, culminating in an output that more clearly signifies the intended emotion.

The core hypothesis guiding these modifications was simple: if the motion generated is sufficiently distinct, then a straightforward subnetwork should be capable of identifying the input emotion based on the motion alone. By optimising the network to reduce emotion loss (EL)—calculated as the mean squared error between the subnetwork's emotion output and the original emotion input—the aim is for the network to produce motions that more faithfully and recognisably embody the specified emotions.