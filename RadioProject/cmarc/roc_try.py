from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the classification data set
from common.io_operations import load_data
from common.utils import load_model

data = load_data(how_many=14)
data = data.astype({'class': str})
print("Number of music entries")
print(data[data['class'] == "music"].shape)
print("Number of no_music entries")
print(data[data['class'] == "no_music"].shape)
# Specify the features of interest and the classes of the target
features = data.columns[:705]

classes = ["music", "no_music"]

# Extract the instances and target

X = data[features]
y = data['class']

# Create the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
from yellowbrick.classifier import ROCAUC
from sklearn.linear_model import LogisticRegression

# model = load_model("models\\random_forest\\rf-10-music-nestimators25.joblib")
model = RandomForestClassifier(n_estimators=25, n_jobs=4, random_state=0, verbose=1)
# Instantiate the visualizer with the classification model
visualizer = ROCAUC(model, classes=classes)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data