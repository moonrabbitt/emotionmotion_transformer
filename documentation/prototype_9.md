# Prototype 9

## Architecture change

ok now that there are more dynamic motions, and the model is not constently repeating itself or getting stuck in transitional motion in a live loop as you can see [here](FILL HERE), but it seems the emotion context of the motion generation is still kinda loss and not very clear. Not as clear as when there wasnt MDN but the model got stuck when it was in a live loop. 

Architecture change - put logits instead of x into emotion_fc2 because x has emotions attached to the end and model seems to somehow be able to learn that it doesnt have to look at the other motions and just use the end 7 numbers to reduce the loss instead. So intead of x I put in the logits into emotion_fc2 (emotion dense layer), with the logic that, for the model to be able to capture emotion context, it should also be able to say what emotion the input is from the generated motion. I also cranked up this emotion loss * 10 so the model really focuses on learning the motion based on emotion context.

## Attention Pooling

WTF IS THIS idk. 