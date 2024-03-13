# Prototype 6

## Refactoring and Code Optimization

At the beginning of this week, I took significant steps to refactor the code from a notebook into several script files. This reorganization was aimed at facilitating easier development and debugging. You can explore the evolution of this process through my git commit history.

Access the full code for Prototype 6 [here](https://github.com/moonrabbitt/emotionmotion_transformer/tree/prototypes_inter).

## Week 12: Enhancing Model Dynamics

This week's focus was on mitigating the static nature of the model by implementing techniques to foster more dynamic outputs. A critical observation was that the model tended to overfit, especially biasing towards stationary movements that are common across all emotions.

### Observing Model Limitations

The model frequently defaulted to intermediate movements like rocking or standing still, evident in its outputs [here](FILL IN HERE). This issue became more pronounced when implementing the model in a live video feed [here](FILL IN HERE). The potential cause of this issue might be the uncoupling of emotion from motion in the live loop, as opposed to validation test sets where they are coupled. This leads the model to favor intermediate movements, resulting in a lack of dynamism.

### Implementing L1/L2 Normalization

To combat overfitting, I experimented with L1 (Lasso) and L2 (Ridge) regularization techniques. These methods add penalties to the model's loss function based on the coefficients' magnitudes, theoretically encouraging it to learn more varied and dynamic patterns. However, this approach led to the model freezing entirely, as shown [here](FILL HERE).

### Penalizing Small Movements

Next, I attempted to penalize small movements by adjusting the loss function to inversely relate to the cumulative delta between frames. This technique, though, resulted in increased jitteriness without significantly enhancing emotionally relevant motions. Even after averaging the cumulative delta over batches, the desired outcome of more dynamic motion was not achieved.

### Exploring Penalization Masks

Another approach was to apply a penalization mask during training for deltas below a certain threshold. The aim was to subtly improve motion without sacrificing smoothness. Unfortunately, this technique also fell short of expectations, as seen [here](FILL HERE).

### Considering Mixture Density Networks (MDN)

Reflecting on the current challenges, I was inspired by the work of Pette et al. (2019) ([source](http://arxiv.org/abs/1907.05297)), who successfully achieved varied and dynamic motion in an LSTM sequencer using MDN. Their success suggests that incorporating MDN might introduce the needed variation in the outputs. However, the differences between LSTM and transformer models in this context are notable. LSTMs are adept at capturing temporal dependencies and might be more suited for sequential tasks like motion prediction. Transformers, while powerful for parallel processing and capturing long-range dependencies, might not be as inherently efficient in this specific task of capturing local patterns, and it simply might be better at regurgitation. This difference might contribute to the current challenges, but an exploration into implementing MDN is still worthwhile.
