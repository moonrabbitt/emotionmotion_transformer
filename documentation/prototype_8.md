# Prototype 8
 
## Week 15

# focussing on loss


 So I had to add and implement a scheduler to schedule when the model should consider only the total MSE losses (keypoints + emotions MSE losses) and when MDN loss should come into effect. I monitored all this using tensorboard

total loss then becomes = MSE(keypoints)+MSE(emotions) + weight*MDN

I did several studies detailed below:
1) Keeping MDN constant at 0.1 or 1 (1 is the full MDN loss weight) throughout - 0.1 - MDN is too crap it doesnt capture the keypoints very well, MDN =1, the MDN loss overpowers the MSE losses leading to the frames not being smooth at all. 

2) Starting after a certain point and slow linear increase: Through observation of losses without MDN I saw that the model loss stops improving at around 40k epochs. So I decided to add the MDN weight after 40k epochs. I found that if the MDN loss weight was initially 0,  when the MDN weight stops being 0, the trained transformer MSE losses then became very erratic and variable and became a lot higher. On the otherhand if the MDN was low at 0.1 at the beginning and increases after 40k epochs, the loss of the transformer is not as erratic, lead to quite good losses. The slow linear increase seems to stop the losses being erratic, so it's better than just increasing in a big step from 0.1 to 1. Waiting a bit before increasing the MDN loss weight allows the model to focus on training the transformer and getting that loss down first before the MDN losses overpowers the training

3) Starting after a certain point and slow exponential increase: Also good but doesnt seem to be as good as linear increase as you can see (FILL HERE)

So I decided to use start after a certain point and slow linear increase. I normally train for around 200k epochs, but with MDN it needs ore training so I train for 300k, so I start MDN loss at 0.1, and start increasing from 0.1 - 1 in a linear ramp over 100k epochs after 50k epochs as you can see below:

PIC HERE

MDN allows more control over generation because I can now control the sampling as well. So I experimented with sampling and wrote different sampling algos.

1) Max Probability sampling : This gave the most smooth result, but is also closest to the transformer output so it is the most static and often is stuck in the same transitional movements like the transformer without MDN

(ADD EQUATION / LOGIC HERE)

2) Next max probability Sampling, controling for sigma (EXPLAIN WTF IS SIGMA) : Next is sampling the next most likely sample, this introduced more noise but more variation as well. Playing around with the function I found that if I / sigma to decrease the variance, I can control how noisy it is by controlling how far to sample from the peak of that gaussian as there is a tradeoff between dynamic motion / noise.
   
3) Random sampling : There's more motion and less standing still defo but it's noisy a f, and also seem to have no emotion context to the motion.

In the end I decided to use next max probability sampling and control the sigma on a curve (cos curve) from 100 to 1 with one being the noisiest. 

This is because I found that if I input noise into the system with an emotion, it lead to more dynamic motion with better emotional context than if I pass a frame from the test input. I think it's because whe it's noisy data it has never seen before, the model focuses on the emotion and doesnt get overly stuck on continuing the previous frame, as you can see [here](FILL HERE). So controlling variance on a curve means that some noisy frames can be passed into the transformer but not too much that the output isnt coherent, so here we can keep the smooth transition between frames whilst going from less to more dynamic motion.
(Maybe rephrase this)