# Prototype 8
 
## Week 14

# Focussing on loss

So I had to add and implement a scheduler to schedule when the model should consider only the total MSE losses (keypoints + emotions MSE losses) and when MDN loss should come into effect. I monitored all this using tensorboard

total loss then becomes = MSE(keypoints)+MSE(emotions) + weight*MDN

I did several studies detailed below:
1) Keeping MDN constant at 0.1 or 1 (1 is the full MDN loss weight) throughout - 0.1 - MDN is too crap it doesnt capture the keypoints very well, MDN =1, the MDN loss overpowers the MSE losses leading to the frames not being smooth at all. 

2) Starting after a certain point and slow linear increase: Through observation of losses without MDN I saw that the model loss stops improving at around 40k epochs. So I decided to add the MDN weight after 40k epochs. I found that if the MDN loss weight was initially 0,  when the MDN weight stops being 0, the trained transformer MSE losses then became very erratic and variable and became a lot higher. On the otherhand if the MDN was low at 0.1 at the beginning and increases after 40k epochs, the loss of the transformer is not as erratic, lead to quite good losses. The slow linear increase seems to stop the losses being erratic, so it's better than just increasing in a big step from 0.1 to 1. Waiting a bit before increasing the MDN loss weight allows the model to focus on training the transformer and getting that loss down first before the MDN losses overpowers the training

3) Starting after a certain point and slow exponential increase: Also good but doesnt seem to be as good as linear increase as you can see (FILL HERE)

So I decided to use start after a certain point and slow linear increase. I normally train for around 200k epochs, but with MDN it needs ore training so I train for 300k, so I start MDN loss at 0.1, and start increasing from 0.1 - 1 in a linear ramp over 100k epochs after 50k epochs.

# Exploring MDN output for better sampling

After seeing that the sampling from the Mixture Density Network (MDN) wasn't as good as I hoped, I took a closer look by analysing and visualising the output of each Gaussian part. I noticed something interesting: the parts with higher π (which shows how likely they are to be picked) created more stagnant motion but contains better human form, but the ones with lower π moved a lot more, but they started to lose their human shape. So, I thought it would be a good idea to try and find a middle ground with my sampling method.

To tackle this, I wrote a new piece of code (see sample -mdn). This time, instead of just picking randomly, I looked for the Gaussian that wasn't just likely but also added some movement to the picture. The idea was to avoid making the output too jittery or just repeating the same movements over and over.

Here’s a brief run-through of what I did in the code:

-Instead of choosing completely at random, I focused on finding the mean of the most probable component but also considered movement by checking which Gaussians showed a decent amount of change.
-To add variety and not just stick to one pattern, I picked indices that showed more movement but weren't just the top movers, making sure there's a bit of unpredictability.
-Then, I mixed the most probable Gaussian with one that adds good movement, aiming for a balance where the output stays true to form but doesn't get boring or too static.
-By adjusting the variance (how spread out the Gaussian is), I hoped to reduce the jitters in the output. The idea was that a smaller variance would mean less wild guesses and a smoother look, but still keeping things interesting.
-This approach was a bit of a balancing act, trying to keep the animation smooth and recognisable while adding enough variation to make it lively and not just a loop of the same actions.






