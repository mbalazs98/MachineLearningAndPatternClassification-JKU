import os

from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def load_data(root_dir=r'f:\SharedData\JKU\GIT\JKU-Computer-Science-Courses\ML-PatternClassification\train',
              data_type='music', how_many=14, last=False):
    """
    This function will load data from all the files 1.<data_type>.arff to <how_many>.<data_type>.arff
    By default will load music training instances from all the files 1..14

    :param root_dir: dir to path where arff files containing data are stored
    :param data_type: str: 'music' or 'speech'
    :param how_many: 'number of files to load' ,
    :return: a pandas dataframe containing loaded data
    """

    dfl = []
    if not last:
        for i in range(1, how_many + 1):
            data = arff.loadarff('{}{}{}.{}.arff'.format(root_dir, os.sep, i, data_type))
            df = pd.DataFrame(data[0])
            dfl.append(df)
    else:
        for i in range(14, 14 - how_many, -1):
            data = arff.loadarff('{}{}{}.{}.arff'.format(root_dir, os.sep, i, data_type))
            df = pd.DataFrame(data[0])
            dfl.append(df)
    df = pd.concat(dfl)
    return df

#
# y = ["music", "no_music", "music", "music", "music", "music", "no_music", "casian"]
# x, xx = pd.factorize(y)
# print(x)
# print(xx)
# # print(MultiLabelBinarizer().fit_transform(y))
