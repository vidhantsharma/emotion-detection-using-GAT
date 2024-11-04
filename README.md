## Facial Emotion Detection using GAT ##

This project will focus on emotion detection. The dataset used is from CK+.

# Model Used - #
1. Image is passed from Resnet18 and a 512 dimension representation is obtained.
2. A multi-headed Graph attnetion network is used on the facial landmarks features. Relative distance features between each landmark with the center point of all the landmarks is taken. A graph connected with nodes of nose, mouth, etc is made and trained using GAT to get a 64 dimension output for each head.
3. Resnet18 output and GATs output are concatinated and fed into a fully connected layer

# How to Use - #
1. Make sure your data follows the structure
        -data
            -emotion
                -train
                -val
                -test
2. Helper function "createDataSplit.py" is provided for doing this. You just need to provide the data path to it which contains images in this structure
        -data
            -emotion
3. use "train.py" to train the model. The checkpoint will be stored by the name of "final_model.pt"
4. The model accuracy can then be tested using "test.py"