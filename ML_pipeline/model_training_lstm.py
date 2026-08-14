import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import pickle

# TRAINING BINARY CLASSIFICATION MODEL

def main(labels):

    dfs = []
    gesture_to_id = {}
    #load csv
    for i, label in enumerate(labels):
        df = pd.read_csv(f"ML_pipeline/ASL_alphabet/{label}.csv")
        dfs.append(df)
        
        #map use numbers to represent classes
        gesture_to_id[f"{label}"] = i
    combined = pd.concat(dfs,ignore_index=True)

    #map labels to new number classifications
    combined["label"] = combined["label"].map(gesture_to_id)

    #take only hand0 8,20,label,clip_id because only landmarks key for J and Z gestures
    #combined = combined.iloc[:, [8,9,20,21,-2,-1]]
    combined = combined.iloc[:, [*range(42), -2, -1]]  # Include all columns except the last two (label and clip_id)

    no_clips = combined['clip_id'].nunique()
    no_labels = combined['label'].nunique()

    X = []
    y = []
    #for each clip in each label
    for i in range(no_labels):
        for j in range(no_clips):
            # Extract features for the current clip and label
            clip_data = combined[(combined['label'] == i) & (combined['clip_id'] == j)]
            X.append(clip_data.iloc[:, :-2].values)  # Append all columns except the last two (label and clip_id)
            y.append(i)  # Append the label for the current clip

    #one-hot encoding y 
    #y = to_categorical(y, num_classes=no_labels)
    #print(y)


 
  
    #combined.to_csv('xxx.csv',index=False)

    #split both csv 70/20/10 train/test/test_2
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,#value not important, as long as is consistent 
        stratify=y

    )
    #print(X_train[0].shape)

    #build the model
    model = Sequential([
        LSTM(64,#64 determines how many params to yse to quantify the sequence(more = higher processingpower, less = reductionist output)
            input_shape=(X_train[0].shape), #27 frames per clip, 4 landmarks features per frame
            return_sequences=False), #return only final hidden state. best for classification tasks
        Dense(32,#32 determines how many params to yse to quantify the sequence(more = higher processingpower, less = reductionist output)
              activation='relu'),#introducing nonlinearity to the model, allowing it to learn more complex patterns in the data
        Dense(no_labels,#no of classes
              activation='softmax')#output layer with softmax activation for multi-class classification. convert final output into class probabilities 

    ])

    model.compile(loss='sparse_categorical_crossentropy',#loss function for multi-class classification tasks, measures the difference between predicted and true class probabilities. categorical c
                  optimizer=Adam(),#default choice optimizer for training deep learning models. adaptive learning rate optimization algorithm that adjusts the learning rate based on the gradients of the loss function
                  metrics=['accuracy']) #use accuracy metric when balanced classification tasks, measures the proportion of correctly classified samples out of the total number of samples
    #print a summary of model architecture, including the number of parameters in each layer and the total number of parameters in the model
    model.summary()

    print(f"X_train shape: {np.array(X_train).shape}")
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=1, validation_data=(X_test, y_test))

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")

    #save model
    model.save("ML_pipeline/models/LSMT_jz_all_landmarks_other.keras") 

if __name__ == '__main__':
    main(['J','Z','other'])
    #['peace','high_five','sixer',"thumbs_up",'f_sign','take_the_l','pinch']