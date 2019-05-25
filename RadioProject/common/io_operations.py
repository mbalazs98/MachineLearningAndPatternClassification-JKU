import os

from scipy.io import arff
import pandas as pd


class DataLoader:
    VALIDATION_FILES_NO = [7, 8, 11]

    @staticmethod
    def load_data():
        """

        :return: two data frames (df) : first one containing the dataset for music/no_music classifier and second one
        containing the dataset for speech/no_speech classifier
        """
        return load_validation_data("music"), load_validation_data("speech")


TRAINING_FILES_NO = [3, 4, 5, 6, 9, 12, 13, 14]
TEST_FILES_NO = [1, 2, 10]
VALIDATION_FILES_NO = [7, 8, 11]


def load_data_from_list_of_files(data_type, list_of_files=None):
    if list_of_files is None:
        list_of_files = TRAINING_FILES_NO
    dfl = []
    for i in list_of_files:
        dfl.append(load_data_from_specific_file_no(data_type=data_type, file_no=i))
    df = pd.concat(dfl)
    return df


def load_training_data(data_type):
    return load_data_from_list_of_files(data_type, TRAINING_FILES_NO)


def load_test_data(data_type):
    """

    :param data_type:
    :return:
    """
    return load_data_from_list_of_files(data_type, TEST_FILES_NO)


def load_validation_data(data_type):
    return load_data_from_list_of_files(data_type, VALIDATION_FILES_NO)


def load_data_from_specific_file_no(
        root_dir=r'f:\SharedData\JKU\GIT\JKU-Computer-Science-Courses\ML-PatternClassification\train',
        data_type='music', file_no=1):
    """
    This function will load data from all the files 1.<data_type>.arff to <how_many>.<data_type>.arff
    By default will load music training instances from all the files 1..14

    :param file_no: file no to load
    :param root_dir: dir to path where arff files containing data are stored
    :param data_type: str: 'music' or 'speech'
    :return: a pandas dataframe containing loaded data
    """

    data = arff.loadarff('{}{}{}.{}.arff'.format(root_dir, os.sep, file_no, data_type))
    df = pd.DataFrame(data[0])
    return df


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

# a = load_validation_data('music')
# print(a.shape())
