from scipy.io import arff
import pandas as pd


def load_data(root_dir=r'f:\SharedData\JKU\GIT\JKU-Computer-Science-Courses\ML-PatternClassification\train',
              data_type='music', how_many=14):
    """
    This function will load data from all the files 1.<data_type>.arff to <how_many>.<data_type>.arff
    By default will load music training instances from all the files 1..14

    :param root_dir: dir to path where arff files containing data are stored
    :param data_type: str: 'music' or 'speech'
    :param how_many: 'number of files to load' ,
    :return: a pandas dataframe containing loaded data
    """

    dfl = []
    for i in range(1, how_many + 1):
        data = arff.loadarff(r'{}.{}.arff'.format(root_dir, data_type))
        df = pd.DataFrame(data[0])
        dfl.append(df)
    df = pd.concat(dfl)
    return df
