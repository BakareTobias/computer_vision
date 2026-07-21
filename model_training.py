import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# TRAINING BINARY CLASSIFICATION MODEL

def main(labels):

    dfs = []
    #load csv
    for label in labels:
        df = pd.read_csv(f"ML_pipeline/datasets/{label}.csv")
        dfs.append(df)
    combined = pd.concat(dfs,ignore_index=True)

    #replace peace sign with 0, high_five with 1
    gesture_to_id = {
        "peace": 0,
        "high_five": 1
    }
    combined["label"] = combined["label"].map(gesture_to_id)


    X = combined.iloc[:, :-1]
    y = combined.iloc[:, -1]


    #split both csv 70/20/10 train/test/test_2
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,#value not important, as long as is consistent 
        stratify=y

    )

    #load into training model 
    #train model?
    #test model


if __name__ == '__main__':
    main(['peace','high_five'])