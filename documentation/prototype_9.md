# Prototype 9


## Architecture Change

- **Dynamic Motion Enhancement:** The model now exhibits more dynamic motions without repeating itself or getting stuck in transitional motion in a live loop, as demonstrated [here](FILL HERE). However, the clarity of emotion context in motion generation is still somewhat ambiguous, particularly in comparison to the pre-MDN era where emotion context was clearer, but the model faced live loop constraints.

- **Emotion Context Integration:** To address this, we've shifted from using `x` to using `emotion_logits` in the `emotion_fc2` layer. The rationale is to compel the model to engage with the full range of motion data rather than simplifying the task by focusing solely on the emotional data at the end of `x`. This change is aimed at enhancing the model's ability to infer the input emotion from the generated motions. Emotion loss was calculated by how far the predicted emotion from the generated motion was from the actual emotion. Logically, this should push the model to generate motion that is more recognisable to a specific emotion (and therefore easier to predict the emoiton from the motion leading to lower loss), rather than common transitional motions. Additionally, the emotion loss has been amplified by a factor of 10, redirecting the model's focus towards learning motions driven by emotional context.

## Attention Pooling

- **Implementation:** Attention pooling was implemented to refine the model's focus on the most relevant features within a given input sequence. This mechanism is critical for models dealing with sequential data, where the relevance of different parts of the sequence can vary significantly.

- **Function: `AttentionPooling`**
  - **Purpose:** The `AttentionPooling` class is designed to compute a weighted average of the input features, where the weights are determined by the importance of each feature in the context of the sequence.
  - **Process:**
    - The input `x` (a tensor with dimensions [B, T, dim]) passes through a linear layer to produce attention scores.
    - These scores are normalized using softmax, ensuring that they sum up to one and reflect the relative importance of each timestep in the sequence.
    - The final output is a weighted average of the input features, scaled by the computed attention scores. This results in a single vector that encapsulates the most significant information from the sequence.

- **Integration in `MotionModel`:**
  - The `AttentionPooling` module is utilized after generating logits from the model's sequence of motions. It pools these logits to focus on the most relevant motion features before they are passed into the `emotion_fc2` layer.
  - This pooling is particularly beneficial for emotion representation, as it allows the model to concentrate on the aspects of the motion sequence that are most indicative of the underlying emotional state.



## Adapting MDN Sampling for Emotion Awareness

Building on the architecture changes and the introduction of emotion logits, a critical aspect of development has been adapting the Mixture Density Network (MDN) sampling process to integrate emotion awareness more effectively. This adaptation is evident in the functions `calculate_dynamic_emotion_scores_individual` and `sample_dynamic_emotion_individual`.

#### Function: `calculate_dynamic_emotion_scores_individual`

- **Purpose:** This function calculates individual scores for each keypoint, considering dynamic movement, noise, and emotional context.
- **Implementation Details:**
  - **Movement Score:** Calculates the difference in position (delta) for each keypoint across frames, then derives a movement score from these deltas, emphasizing the dynamic aspect of motion.
  - **Noise Penalty:** Applies a noise penalty based on the standard deviation (`sigma`) of the Gaussian components, weighted by `k`. This accounts for the variability or uncertainty in keypoint positioning.
  - **Emotion Score:** The emotion logits are expanded and used to calculate an emotion score for each keypoint. This score prioritizes the dominant emotion while penalizing neutral expressions (indexed by `neutral_index`). This ensures that keypoints are scored higher when they are associated with stronger emotional expressions.

#### Function: `sample_dynamic_emotion_individual`

- **Purpose:** This function samples keypoints for the next frame, focusing on individual keypoints with emotion awareness.
- **Implementation Highlights:**
  - **Contextual Scoring:** It first calculates scores for the next frame using `calculate_dynamic_emotion_scores_individual`, incorporating the last frames as context.
  - **MDN Parameter Selection:** The function then isolates the MDN parameters (pi, mu, sigma) for the last timestep.
  - **Score Combination:** Scores are combined with the logarithm of the mixture component weights (`pi`), balancing the physical dynamics with the emotional context.
  - **Sampling Process:**
    - For each keypoint (`o`), it creates a categorical distribution based on the combined scores.
    - It probabilistically selects a Gaussian component for each keypoint.
    - Selected parameters (mean and standard deviation) from the chosen Gaussian component are used to sample the keypoint's position for the next frame.
  - **Normalization:** The `selected_sigma_o` is normalized by `variance_div` to manage the variability in sampling, ensuring that the sampled motions are not too erratic.
