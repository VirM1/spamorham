import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

cv = CountVectorizer()
clf = MultinomialNB()

def processML(chunk):
    X_train, X_test, y_train, y_test = returnValues(chunk)
    clf.partial_fit(X_train, y_train, classes=[True,False])
    print("Current accuracy of Model:", clf.score(X_train, y_train) * 100, "%")

def returnValues(chunk):
    df_z = chunk["message"]
    df_y = chunk["label"]
    corpus = df_z
    X = cv.transform(corpus.values.astype('U'))
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.1, random_state=0)
    return X_train, X_test, y_train, y_test

chunksize = 10000
filename = "C:\\Users\\virgi\\Documents\\ML - Projects\\emailSpamV1\\refinedDatasets\\out_training.csv"
sample_size = 10000

# Fit CountVectorizer on a sample of the data
sample_data = pd.read_csv(filename, header=0, quotechar='"', encoding="ISO-8859-15", nrows=sample_size)
cv.fit_transform(sample_data["message"].values.astype('U'))

with pd.read_csv(filename, chunksize=chunksize, header=0, quotechar='"', encoding="ISO-8859-15") as reader:
    finalized_filename = 'finalized_model.sav'
    for index, chunk in enumerate(reader):
        processML(chunk)

    joblib.dump(clf, finalized_filename)

comment = ["Hello! Impotent? Revive your virility with the brand-new PPHard pills, priced at just 50 cents. Doctors hate them!"]
vect = cv.transform(comment).toarray()
print(clf.predict(vect))
