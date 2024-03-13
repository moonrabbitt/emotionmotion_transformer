# Prototype 7

## Week 13
### Considering Mixture Density Networks (MDN)

Incorporating Mixture Density Networks (MDN) into the transformer architecture, inspired by the methodology of Pette et al., marked a significant milestone in the project. Despite the successful integration of MDN, the outcomes have been somewhat disappointing. The animations exhibit a jittery quality during both training and testing phases. Furthermore, when the model is applied in a live feedback loop, it tends to fall into repetitive patterns, lacking the desired variability and fluidity.


# Exploring sampling
MDN allows more control over generation because I can now control the sampling as well. So I experimented with sampling and wrote different sampling algorithms.

1) Max Probability sampling : This gave the most smooth result, but is also closest to the transformer output so it is the most static and often is stuck in the same transitional movements like the transformer without MDN


2) Next max probability Sampling, controling for sigma (EXPLAIN WTF IS SIGMA) : Next is sampling the next most likely sample, this introduced more noise but more variation as well. Playing around with the function I found that if I / sigma to decrease the variance, I can control how noisy it is by controlling how far to sample from the peak of that gaussian as there is a tradeoff between dynamic motion / noise.
   
3) Random sampling : There's more motion and less standing still defo but it's noisy a f, and also seem to have no emotion context to the motion.

I landed on using next max probability sampling with sigma control via a cosine curve from 100 to 1, where 1 is the peak noise level. 

This choice stemmed from a neat find: mixing noise with an emotion into the mix actually led to animations that felt more alive and emotionally resonant, unlike feeding it a frame straight from the test set. It seems the model hones in on the emotion better with fresh, noisy data—data it hasn't seen before—without getting hung up on replicating the previous frame too closely. By dialing the variance up and down on this curve, I could let some noise slip into the transformer without muddying the waters too much. This way, we're navigating from calm to stormy waters and back, ensuring our animation transitions smoothly while gradually ramping up the drama.