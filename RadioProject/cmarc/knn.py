import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from common.io_operations import load_data


def store_model(model, path):
    joblib.dump(model, path)


def get_test_and_train_data(how_many, training_percentage, features_nr=705, last=False, data_type="music"):
    print("Start loading data from {} files...".format(how_many))
    df = load_data(how_many=14, last=last, data_type=data_type)

    print("Done loading data.")
    print("Start splitting data into training {}, test {}".format(training_percentage, 1 - training_percentage))

    np.random.seed(0)
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= training_percentage
    df = df.astype({'class': str})
    print("Number of music entries")
    print(df[df['class'] == "music"].shape)
    print("Number of no_music entries")
    print(df[df['class'] == "no_music"].shape)
    train, test = df[df['is_train'] == True], df[df['is_train'] == False]

    # Show the number of observations for the test and training dataframes
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    # Create a list of the feature column's names
    features = df.columns[:features_nr]

    return train, test, features


from sklearn.neighbors import KNeighborsClassifier

train, test, features = get_test_and_train_data(14, 0.75)

knn = KNeighborsClassifier(n_jobs=4)
knn.fit(train[features], train["class"])
p = knn.predict(test[features])

acc = accuracy_score(test["class"], p)
print("Accuracy on test set {}".format(acc))

store_model(knn, r'models\knn1-all-rob.joblib')
