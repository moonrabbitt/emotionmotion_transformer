# Prototype 4

## Week 9

The code for this prototype is available [here](notebooks/prototypes/inter-prototype-4.ipynb).

This week I focussed on adding the relationship between motion and emotion to the transformer. The transformer's input structure was updated as [Batch, Time, Keypoints+emotion], where 'Batch' is the number of videos, 'Time' refers to the number of frames, and 'Keypoints' includes 25 points, each with x and y coordinates, totalling 50 numbers. Additionally, an emotion vector is appended—a one-hot encoding of seven emotions: Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Neutral. The transformer's output mirrors its input structure. The emotion was added to the end of every frame with thoughts that it will allow some flexibility during traning for emotions to change, and allow an emotional story to develop by changing the emotions throughout the video. The results of integrating emotion can be seen [here](https://drive.google.com/file/d/1udf-rDy86NtPzTLwH3QNBdkUi4ZS3oEn/view?usp=sharing).

I also worked on several data preprocessing to integrate motion with emotion such as normalizing the data and stratifying the train, test, and validation sets by emotions to ensure balanced representation and avoid bias toward any specific emotion. 