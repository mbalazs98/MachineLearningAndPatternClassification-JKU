import os

import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from common.utils import store_model, load_model
from common.io_operations import load_data
from sklearn.metrics import accuracy_score


def create_a_rf_classifier(n_estimators=10, max_depth=None, min_samples_leaf=1):
    # Create a random forest Classifier. By convention, clf means 'Classifier'
    return RandomForestClassifier(n_estimators=n_estimators, n_jobs=8, random_state=0, verbose=1, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf)


def predict(model, data, features):
    return model.predict(data[features])


def compute_accuracy(data, pred, data_type="test"):
    correct = 0
    total = len(pred)
    for i in range(len(pred)):
        # print(test[i:i+1]["class"][0:1])
        # print(type(test[i:i+1]["class"][0:1]))
        if pred[i] == data[i:i + 1]["class"].values[0]:
            correct += 1
    print("Accuracy on {} = {}".format(data_type, correct / total))


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


def train_model_using_data_from_file(model, features, train):
    model.fit(train[features], train['class'])
    return model


def round_01():
    how_many = 10
    model_file_path = "models\\random_forest\\rf-01-all-music.joblib"
    new_model = True
    models_root_dir = "models\\random_forest"
    model_name = "rf-01-all-music.joblib"
    if new_model and model_name is None or model_name == "":
        raise ValueError("Provide a model name")
    elif new_model:
        model = create_a_rf_classifier()
    else:
        model = load_model(model_file_path)
    train, test, features = get_test_and_train_data(how_many, training_percentage=0.75)
    model = train_model_using_data_from_file(model, features, train)

    pred = predict(model, test, features)
    acc = accuracy_score(test["class"], pred)
    print("Accuracy on test set {}".format(acc))

    pred = predict(model, train, features)
    acc = accuracy_score(train["class"], pred)
    print("Accuracy on training set {}".format(acc))

    if new_model:
        store_model(model, os.path.join(models_root_dir, model_name))


def test_round_1():
    model_file_path = "models\\random_forest\\rf-01-all-music-music.joblib"
    model = load_model(model_file_path)
    data = load_data(how_many=4, last=True)
    data = data.astype({'class': str})
    features = data.columns[:705]

    pred = predict(model, data, features)
    acc = accuracy_score(data["class"], pred)
    print("Accuracy on validation set {}".format(acc))


def round_01_speech():
    how_many = 10
    model_file_path = "models\\random_forest\\rf-01-all-speech.joblib"
    new_model = True
    models_root_dir = "models\\random_forest"
    model_name = "rf-01-all-speech.joblib"
    if new_model and model_name is None or model_name == "":
        raise ValueError("Provide a model name")
    elif new_model:
        model = create_a_rf_classifier()
    else:
        model = load_model(model_file_path)
    train, test, features = get_test_and_train_data(how_many, training_percentage=0.75, data_type="speech",
                                                    features_nr=103)
    model = train_model_using_data_from_file(model, features, train)

    pred = predict(model, test, features)
    acc = accuracy_score(test["class"], pred)
    print("Accuracy on test set {}".format(acc))

    pred = predict(model, train, features)
    acc = accuracy_score(train["class"], pred)
    print("Accuracy on training set {}".format(acc))

    if new_model:
        store_model(model, os.path.join(models_root_dir, model_name))


def test_round_1_speech():
    model_file_path = "models\\random_forest\\rf-01-all-speech.joblib"
    model = load_model(model_file_path)
    data = load_data(how_many=4, last=True, data_type="speech")
    data = data.astype({'class': str})
    features = data.columns[:103]

    pred = predict(model, data, features)
    acc = accuracy_score(data["class"], pred)
    print("Accuracy on validation set {}".format(acc))


def roc(model, data_type="music", features_nr=705):
    classes = ["{}".format(data_type), "no_{}".format(data_type)]
    from yellowbrick.classifier import ROCAUC

    data = load_data(how_many=4, last=True, data_type=data_type)
    data = data.astype({'class': str})

    features = data.columns[:features_nr]
    X = data[features]
    y = data["class"]

    # Instantiate the visualizer with the classification model
    visualizer = ROCAUC(model, classes=classes)

    visualizer.score(X, y)  # Evaluate the model on the test data
    g = visualizer.poof()  # Draw/show/poof the data


def roc_music_rf():
    model = load_model("models\\random_forest\\rf-01-all-music.joblib")
    roc(model)


def roc_speech_rf():
    model = load_model("models\\random_forest\\rf-01-all-speech.joblib")
    roc(model, features_nr=103, data_type="speech")


def get_graph_for_n_estimators(data_type="music"):
    models_root_dir = "models\\random_forest"
    accuracy_test = []
    accuracy_training = []
    n_estimators = []
    features_no = 705
    if data_type == "speech":
        features_no = 103
    train, test, features = get_test_and_train_data(how_many=14, training_percentage=0.75, data_type=data_type,
                                                    features_nr=features_no)
    for i in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200]:
        print("Training rf with {} estimators...".format(i))
        n_estimators.append(i)
        model = create_a_rf_classifier(i)
        model_name = "rf-10-{}-nestimators{}.joblib".format(data_type, i)

        model = train_model_using_data_from_file(model, features, train)

        pred = predict(model, test, features)
        acc = accuracy_score(test["class"], pred)
        print("Accuracy on test set {}".format(acc))
        accuracy_test.append(acc)

        pred = predict(model, train, features)
        acc = accuracy_score(train["class"], pred)
        print("Accuracy on training set {}".format(acc))
        accuracy_training.append(acc)

        store_model(model, os.path.join(models_root_dir, model_name))

    plot_vs_n_estim(accuracy_test, accuracy_training, n_estimators, ptitle="Speech Data")


def get_graph_for_max_depth(data_type="music"):
    models_root_dir = "models\\random_forest"
    accuracy_test = []
    accuracy_training = []
    n_estimators = []
    features_no = 705
    if data_type == "speech":
        features_no = 103
    train, test, features = get_test_and_train_data(how_many=14, training_percentage=0.75, data_type=data_type,
                                                    features_nr=features_no)
    for max_depth in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]:
        print("Training rf with max depth {}...".format(max_depth))
        n_estimators.append(max_depth)
        model = create_a_rf_classifier(max_depth=max_depth, n_estimators=100)
        model_name = "rf-10-{}-nestimators_100-max_depth{}.joblib".format(data_type, max_depth)

        model = train_model_using_data_from_file(model, features, train)

        pred = predict(model, test, features)
        acc = accuracy_score(test["class"], pred)
        print("Accuracy on test set {}".format(acc))
        accuracy_test.append(acc)

        pred = predict(model, train, features)
        acc = accuracy_score(train["class"], pred)
        print("Accuracy on training set {}".format(acc))
        accuracy_training.append(acc)

        store_model(model, os.path.join(models_root_dir, model_name))

    plot_vs_sth(accuracy_test, accuracy_training, n_estimators, ptitle="{} Data".format(data_type),
                plot_type="Trees Max Depth")


def get_graph_for_min_samples_leaf(data_type="music"):
    models_root_dir = "models\\random_forest"
    accuracy_test = []
    accuracy_training = []
    n_estimators = []
    features_no = 705
    if data_type == "speech":
        features_no = 103
    train, test, features = get_test_and_train_data(how_many=14, training_percentage=0.75, data_type=data_type,
                                                    features_nr=features_no)
    for min_samples_leaf in [2, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 350, 500]:
        print("Training rf with min samples leaf {}...".format(min_samples_leaf))
        n_estimators.append(min_samples_leaf)
        model = create_a_rf_classifier(min_samples_leaf=min_samples_leaf, n_estimators=100)
        model_name = "rf-14-{}-nestimators_100-min_samples_leaf{}.joblib".format(data_type, min_samples_leaf)

        model = train_model_using_data_from_file(model, features, train)

        pred = predict(model, test, features)
        acc = accuracy_score(test["class"], pred)
        print("Accuracy on test set {}".format(acc))
        accuracy_test.append(acc)

        pred = predict(model, train, features)
        acc = accuracy_score(train["class"], pred)
        print("Accuracy on training set {}".format(acc))
        accuracy_training.append(acc)

        store_model(model, os.path.join(models_root_dir, model_name))

    plot_vs_sth(accuracy_test, accuracy_training, n_estimators, ptitle="{} Data".format(data_type),
                plot_type="Min Samples Leaf")


def plot_vs_sth(m_accuracy_test, m_accuracy_training, m_x, ptitle="Music data",
                plot_type="Number of Estimators"):
    print("{} - {}                    : {}".format(ptitle, plot_type, m_x))
    print("{} - Accuracy test         : {}".format(ptitle, m_accuracy_test))
    print("{} - Accuracy training     : {}".format(ptitle, m_accuracy_training))
    import matplotlib.pyplot as plt
    plt.plot(m_x, m_accuracy_test, "-", label="{} - Accuracy on test".format(ptitle))
    plt.plot(m_x, m_accuracy_training, label="{} - Accuracy on training".format(ptitle))
    plt.ylabel('Accuracy')
    plt.xlabel(plot_type)
    plt.legend()
    plt.title(ptitle)
    plt.show()


def plot_vs_n_estim(m_accuracy_test, m_accuracy_training, m_n_estimators, ptitle="Music data"):
    print("{} - Estimators nr         : {}".format(ptitle, m_n_estimators))
    print("{} - Accuracy test         : {}".format(ptitle, m_accuracy_test))
    print("{} - Accuracy training     : {}".format(ptitle, m_accuracy_training))
    import matplotlib.pyplot as plt
    plt.plot(m_n_estimators, m_accuracy_test, "-", label="{} - Accuracy on test".format(ptitle))
    plt.plot(m_n_estimators, m_accuracy_training, label="{} - Accuracy on training".format(ptitle))
    plt.ylabel('Accuracy')
    plt.xlabel('Number of estimators')
    plt.legend()
    plt.title(ptitle)
    plt.show()


def combined_plot_vs_n_estim(m_accuracy_test, m_accuracy_training, m_n_estimators, s_accuracy_test, s_accuracy_training,
                             s_n_estimators, ptitle):
    print("Music - Estimators nr         : {}".format(m_n_estimators))
    print("Music - Accuracy test         : {}".format(m_accuracy_test))
    print("Music - Accuracy training     : {}".format(m_accuracy_training))
    print("Speech - Estimators nr         : {}".format(s_n_estimators))
    print("Speech - Accuracy test         : {}".format(s_accuracy_test))
    print("Speech - Accuracy training     : {}".format(s_accuracy_training))
    import matplotlib.pyplot as plt
    plt.plot(m_n_estimators, m_accuracy_test, "-", label="{} - Accuracy on test".format("Music"))
    plt.plot(m_n_estimators, m_accuracy_training, label="{} - Accuracy on training".format("Music"))
    plt.plot(s_n_estimators, s_accuracy_test, "-", label="{} - Accuracy on test".format("Speech"))
    plt.plot(s_n_estimators, s_accuracy_training, label="{} - Accuracy on training".format("Speech"))

    plt.ylabel('Accuracy')
    plt.xlabel('Number of estimators')
    plt.legend()
    plt.title(ptitle)
    plt.show()


def combined_plot_vs_sth(m_accuracy_test, m_accuracy_training, m_x, s_accuracy_test, s_accuracy_training,
                         s_x, ptitle, plot_type="Number of Estimators"):
    print("Music - {}                     : {}".format(plot_type, m_x))
    print("Music - Accuracy test          : {}".format(m_accuracy_test))
    print("Music - Accuracy training      : {}".format(m_accuracy_training))
    print("Speech - {}                    : {}".format(plot_type, s_x))
    print("Speech - Accuracy test         : {}".format(s_accuracy_test))
    print("Speech - Accuracy training     : {}".format(s_accuracy_training))
    import matplotlib.pyplot as plt
    plt.plot(m_x, m_accuracy_test, "-", label="{} - Accuracy on test".format("Music"))
    plt.plot(m_x, m_accuracy_training, label="{} - Accuracy on training".format("Music"))
    plt.plot(s_x, s_accuracy_test, "-", label="{} - Accuracy on test".format("Speech"))
    plt.plot(s_x, s_accuracy_training, label="{} - Accuracy on training".format("Speech"))

    plt.ylabel('Accuracy')
    plt.xlabel(plot_type)
    plt.legend()
    plt.title(ptitle)
    plt.show()


def combined_plot_vs_n_estim_call():
    combined_plot_vs_n_estim(
        s_accuracy_test=[0.8596891126096179, 0.9046443156994967, 0.9140785766296261, 0.9181625352201855,
                         0.9200620508437015, 0.9216608098268275, 0.9223414695919208, 0.922848007091525,
                         0.9224997625605471, 0.9235128375597556, 0.9236394719346567, 0.9243676195903378,
                         0.9245259125589641, 0.92492164498053],
        s_accuracy_training=[0.9476462007691288, 0.9889457631613844, 0.9935552314016709, 0.9976183530035805,
                             0.9980480042434691, 0.9990770454846837, 0.9992043495557619, 0.9995650444238164,
                             0.9996393051319453, 0.9998249569022676, 0.9998514785837422, 0.9999522609733457,
                             0.9999734783185253, 0.9999734783185253],
        s_n_estimators=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200],
        m_n_estimators=[1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200],
        m_accuracy_test=[0.93441922309811, 0.9713014847880457, 0.9758444929876214, 0.9811947953271916,
                         0.9810998195460158, 0.9831417988412955, 0.9830468230601197, 0.9843448254028556,
                         0.9844239718871688, 0.9850096558710862, 0.9845980941526578, 0.9856744863393168,
                         0.9856586570424541, 0.9863393168075474],
        m_accuracy_training=[0.9761517040180347, 0.9975069619413871, 0.998992176103965, 0.9998143482296777,
                             0.9998886089378066, 0.9999522609733457, 0.9999734783185253, 0.9999840869911152,
                             0.9999840869911152, 0.999994695663705, 0.9999893913274102, 0.999994695663705,
                             0.999994695663705, 1.0],
        ptitle="Random Forest classifier - Accuracy for different number of estimators"
    )


def combined_plot_vs_sth_call():
    combined_plot_vs_sth(
        s_accuracy_test=[0.9245259125589641, 0.9230379586538766, 0.9219140785766297, 0.9209010035774211,
                         0.9184791211574382, 0.9147275778009941, 0.9128597207712034, 0.9114034254598411,
                         0.9102162281951436, 0.9085224934308418, 0.9078576629626112, 0.905752366479881,
                         0.9044068762465571, 0.9021274574983379, 0.9004337227340361],
        s_accuracy_training=[0.9999734783185253, 0.9911205410423021, 0.9694045882508951, 0.9623339079697653,
                             0.9437634266012466, 0.9302744994032621, 0.9239517305397162, 0.9203606948680546,
                             0.9178676568094417, 0.9140750563585731, 0.9116138443177297, 0.908500198912611,
                             0.9063307253679883, 0.9031481235910357, 0.9008354329664501],
        s_x=[1, 2, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 350, 500],
        m_x=[1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300, 400,
             500],
        m_accuracy_test=[0.9856586570424541, 0.9828093836071802, 0.9804983062652357, 0.9787729129072087,
                         0.976683445721341, 0.9703358976794251, 0.9662202804951404, 0.963687592997119,
                         0.9612815398739988, 0.9593820242504828, 0.9577199480799062, 0.9564061164403077,
                         0.9548231867540444, 0.9543166492544402, 0.9527970367556273, 0.9510874726944629,
                         0.9498527875391775, 0.9485706144933042, 0.9474783930097825, 0.9454364137145028,
                         0.9440117769968658, 0.9429512141070694, 0.941542406686295, 0.9401335992655206,
                         0.9393263051255263, 0.9374901066894609, 0.9356064203628075],
        m_accuracy_training=[0.999994695663705, 0.9998833046015118, 0.9993051319453653, 0.9983238297308049,
                             0.9966264421164301, 0.9884100251955974, 0.9823577774830924, 0.9778703089775892,
                             0.9741944039252088, 0.9716164964858772, 0.9690545020554303, 0.9668585068293329,
                             0.9651027715157141, 0.9633947752287495, 0.9611987800026521, 0.9584829598196526,
                             0.9571568757459223, 0.9555867922026257, 0.9539424479512001, 0.9510197586526986,
                             0.9487388940458825, 0.9473756796180878, 0.9462087256332051, 0.9441771648322503,
                             0.9430420368651372, 0.940639172523538, 0.9387083941121868],
        ptitle="Random Forest classifier - Accuracy for different values for min samples on a leaf(100 estimators used)",
        plot_type="Minimum number of samples on a leaf"

    )


def plot_vs_min_sample_leaf_music():
    m_accuracy_test = [
        0.9856586570424541, 0.9828093836071802, 0.9804983062652357, 0.9787729129072087, 0.976683445721341,
        0.9703358976794251, 0.9662202804951404, 0.963687592997119, 0.9612815398739988, 0.9593820242504828,
        0.9577199480799062, 0.9564061164403077, 0.9548231867540444, 0.9543166492544402, 0.9527970367556273,
        0.9510874726944629, 0.9498527875391775, 0.9485706144933042, 0.9474783930097825, 0.9454364137145028,
        0.9440117769968658, 0.9429512141070694, 0.941542406686295, 0.9401335992655206, 0.9393263051255263,
        0.9374901066894609, 0.9356064203628075]
    m_accuracy_training = [0.999994695663705, 0.9998833046015118, 0.9993051319453653, 0.9983238297308049,
                           0.9966264421164301, 0.9884100251955974, 0.9823577774830924, 0.9778703089775892,
                           0.9741944039252088, 0.9716164964858772, 0.9690545020554303, 0.9668585068293329,
                           0.9651027715157141, 0.9633947752287495, 0.9611987800026521, 0.9584829598196526,
                           0.9571568757459223, 0.9555867922026257, 0.9539424479512001, 0.9510197586526986,
                           0.9487388940458825, 0.9473756796180878, 0.9462087256332051, 0.9441771648322503,
                           0.9430420368651372, 0.940639172523538, 0.9387083941121868]
    m_x = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250,
           300, 400, 500]
    plot_vs_sth(m_accuracy_test=m_accuracy_test[::-1],
                m_accuracy_training=m_accuracy_training[::-1],
                m_x=m_x[::-1],
                ptitle="Music Data", plot_type="Min Samples Leaf")


def select_best_value_for_estim():
    s_accuracy_test = [0.8596891126096179, 0.9046443156994967, 0.9140785766296261, 0.9181625352201855,
                       0.9200620508437015, 0.9216608098268275, 0.9223414695919208, 0.922848007091525,
                       0.9224997625605471, 0.9235128375597556, 0.9236394719346567, 0.9243676195903378,
                       0.9245259125589641, 0.92492164498053]
    s_accuracy_training = [0.9476462007691288, 0.9889457631613844, 0.9935552314016709, 0.9976183530035805,
                           0.9980480042434691, 0.9990770454846837, 0.9992043495557619, 0.9995650444238164,
                           0.9996393051319453, 0.9998249569022676, 0.9998514785837422, 0.9999522609733457,
                           0.9999734783185253, 0.9999734783185253]
    s_n_estimators = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200]
    m_n_estimators = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200]
    m_accuracy_test = [0.93441922309811, 0.9713014847880457, 0.9758444929876214, 0.9811947953271916,
                       0.9810998195460158, 0.9831417988412955, 0.9830468230601197, 0.9843448254028556,
                       0.9844239718871688, 0.9850096558710862, 0.9845980941526578, 0.9856744863393168,
                       0.9856586570424541, 0.9863393168075474]
    m_accuracy_training = [0.9761517040180347, 0.9975069619413871, 0.998992176103965, 0.9998143482296777,
                           0.9998886089378066, 0.9999522609733457, 0.9999734783185253, 0.9999840869911152,
                           0.9999840869911152, 0.999994695663705, 0.9999893913274102, 0.999994695663705,
                           0.999994695663705, 1.0]

    for i in range(len(m_n_estimators)):
        print("Music Difference train-test for {} estimators is {}".format(m_n_estimators[i],
                                                                           m_accuracy_training[i] - m_accuracy_test[i]))

    for i in range(len(s_n_estimators)):
        print("Speech Difference train-test for {} estimators is {}".format(s_n_estimators[i],
                                                                            s_accuracy_training[i] - s_accuracy_test[
                                                                                i]))


def select_best_value_depth():
    s_accuracy_test = [0.8967613258619053, 0.9098046664767151, 0.9187323899072403, 0.9220407129515307,
                       0.9237186184189699, 0.9244151074809257, 0.9237186184189699, 0.9245259125589641,
                       0.9238294234970083, 0.9235919840440687, 0.9243201316997499, 0.9245100832621015,
                       0.9243359609966125, 0.9247000348244531, 0.9245259125589641]
    s_accuracy_training = [0.8974300490651107, 0.9212942580559608, 0.9586102638907307, 0.9794828272112452,
                           0.9887388940458826, 0.9936719268001591, 0.9972258321177563, 0.9990505238032091,
                           0.9998726959289219, 0.9999734783185253, 0.9999734783185253, 0.9999734783185253,
                           0.9999734783185253, 0.9999734783185253, 0.9999734783185253]
    s_depths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    m_depths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    m_accuracy_test = [0.9216924684205527, 0.9439009719188274, 0.9639566910437838, 0.9748947351758634,
                       0.980007598062494, 0.9818121379048342, 0.9831259695444329, 0.9841865324342293, 0.984281508215405,
                       0.9845506062620698, 0.9852154367303004, 0.9855953398550037, 0.9859910722765695,
                       0.9854687054801026, 0.9856586570424541]
    m_accuracy_training = [0.9242434690359369, 0.9508765415727357, 0.9811696061530301, 0.9900119347566636,
                           0.995825487335897, 0.9978199177827874, 0.9988913937143615, 0.9994165230075587,
                           0.9996764354860098, 0.9998833046015118, 0.9999893913274102, 0.999994695663705,
                           0.999994695663705, 0.999994695663705, 0.999994695663705]

    for i in range(len(m_depths)):
        print("Music Difference train-test for {} depth is {}".format(m_depths[i],
                                                                      m_accuracy_training[i] - m_accuracy_test[i]))

    for i in range(len(s_depths)):
        print("Speech Difference train-test for {} depth is {}".format(s_depths[i],
                                                                       s_accuracy_training[i] - s_accuracy_test[
                                                                           i]))


def select_best_value_min_samples_leaf():
    s_accuracy_test = [0.9245259125589641, 0.9230379586538766, 0.9219140785766297, 0.9209010035774211,
                       0.9184791211574382, 0.9147275778009941, 0.9128597207712034, 0.9114034254598411,
                       0.9102162281951436, 0.9085224934308418, 0.9078576629626112, 0.905752366479881,
                       0.9044068762465571, 0.9021274574983379, 0.9004337227340361]
    s_accuracy_training = [0.9999734783185253, 0.9911205410423021, 0.9694045882508951, 0.9623339079697653,
                           0.9437634266012466, 0.9302744994032621, 0.9239517305397162, 0.9203606948680546,
                           0.9178676568094417, 0.9140750563585731, 0.9116138443177297, 0.908500198912611,
                           0.9063307253679883, 0.9031481235910357, 0.9008354329664501]

    s_min_samples = [1, 2, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 350, 500]
    m_min_samples = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250,
                     300,
                     400, 500]
    m_accuracy_test = [0.9856586570424541, 0.9828093836071802, 0.9804983062652357, 0.9787729129072087,
                       0.976683445721341, 0.9703358976794251, 0.9662202804951404, 0.963687592997119, 0.9612815398739988,
                       0.9593820242504828, 0.9577199480799062, 0.9564061164403077, 0.9548231867540444,
                       0.9543166492544402, 0.9527970367556273, 0.9510874726944629, 0.9498527875391775,
                       0.9485706144933042, 0.9474783930097825, 0.9454364137145028, 0.9440117769968658,
                       0.9429512141070694, 0.941542406686295, 0.9401335992655206, 0.9393263051255263,
                       0.9374901066894609, 0.9356064203628075]
    m_accuracy_training = [0.999994695663705, 0.9998833046015118, 0.9993051319453653, 0.9983238297308049,
                           0.9966264421164301, 0.9884100251955974, 0.9823577774830924, 0.9778703089775892,
                           0.9741944039252088, 0.9716164964858772, 0.9690545020554303, 0.9668585068293329,
                           0.9651027715157141, 0.9633947752287495, 0.9611987800026521, 0.9584829598196526,
                           0.9571568757459223, 0.9555867922026257, 0.9539424479512001, 0.9510197586526986,
                           0.9487388940458825, 0.9473756796180878, 0.9462087256332051, 0.9441771648322503,
                           0.9430420368651372, 0.940639172523538, 0.9387083941121868]

    for i in range(len(m_min_samples)):
        print("Music Difference train-test for {} samples is {}".format(m_min_samples[i],
                                                                        (m_accuracy_training[i] - m_accuracy_test[
                                                                            i]) * 100))
        # if m_min_samples[i] == 70:
        print("Accuracy on test: {}".format(m_accuracy_test[i]))
        print("Accuracy on training: {}".format(m_accuracy_training[i]))

    for i in range(len(s_min_samples)):
        print("Speech Difference train-test for {} samples is {}".format(s_min_samples[i],
                                                                         (s_accuracy_training[i] - s_accuracy_test[
                                                                             i]) * 100))
        if s_min_samples[i] == 100:
            print("Accuracy on test: {}".format(s_accuracy_test[i]))
            print("Accuracy on training: {}".format(s_accuracy_training[i]))


def how_many_sampels_per_cat(data_type):
    df = load_data(how_many=14, data_type=data_type)

    print("Done loading data.")

    df = df.astype({'class': str})
    print("Number of music entries")
    print(df[df['class'] == "b'{}'".format(data_type)].shape)
    print("Number of no_music entries")
    print(df[df['class'] == "b'no_{}'".format(data_type)].shape)


if __name__ == "__main__":
    # get_graph_for_min_samples_leaf("speech")
    # select_best_value_min_samples_leaf()

    # how_many_sampels_per_cat("speech")
    get_test_and_train_data(1, 0.3)

    # plot_vs_min_sample_leaf_music()
    # get_graph_for_max_depth("speech")
    # get_graph_for_n_estimators(data_type="speech")
    # combined_plot_vs_n_estim_call()
    # round_01_speech()
    # test_round_1_speech()
    # how_many = 14
    # model_file_path = "models\\rf-all.joblib"
    # new_model = False
    # models_root_dir = "models\\random_forest"
    # model_name = ""
    # if new_model and model_name is None or model_name == "":
    #     raise ValueError("Provide a model name")
    # model = load_model(model_file_path)
    # train, test, features = get_test_and_train_data(how_many, training_percentage=0.75)
    # model = train_model_using_data_from_file(model, features, train)
    # pred = predict(model, test, features)
    # compute_accuracy(test, pred, "test")
    # pred = predict(model, train, features)
    # compute_accuracy(test, pred, "train")
    # if new_model:
    #     store_model(model, os.path.join(models_root_dir, model_name))
