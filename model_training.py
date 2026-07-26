import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# TRAINING BINARY CLASSIFICATION MODEL

def main(labels):

    dfs = []
    gesture_to_id = {}
    #load csv
    for i, label in enumerate(labels):
        df = pd.read_csv(f"ML_pipeline/ASL/{label}.csv")
        dfs.append(df)
        
        #map use numbers to represent classes
        gesture_to_id[f"{label}"] = i
    combined = pd.concat(dfs,ignore_index=True)

    #map labels to new number classifications
    combined["label"] = combined["label"].map(gesture_to_id)


    X = combined.iloc[:, :-1]
    y = combined.iloc[:, -1]
    #combined.to_csv('xxx.csv',index=False)

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
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    #test model
    y_pred = model.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print('Confusion matrix')
    print(cnf_matrix)

    print('Precision:', metrics.precision_score(y_test, y_pred,average=None))
    print('Recall:', metrics.recall_score(y_test, y_pred,average=None))
    print('F1 Score:', metrics.f1_score(y_test, y_pred,average=None))

    #save model
    with open(f'ML_pipeline/models/random_forest_as_thumbsup_pinch.pkl', 'wb') as f:
        pickle.dump(model, f) 

if __name__ == '__main__':
    main(['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','R','S','thumbs_up','pinch'])
    #['peace','high_five','sixer',"thumbs_up",'f_sign','take_the_l','pinch']