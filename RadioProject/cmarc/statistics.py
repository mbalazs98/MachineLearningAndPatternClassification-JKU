import copy
import math
import os
import sys

from scipy.io import arff
import pandas as pd

"""

-------------------> MUSIC <-------------------

    1.music.arff music - 17431; no_music - 462
    2.music.arff music - 16877; no_music - 339
    3.music.arff music - 12393; no_music - 5568
    4.music.arff music - 16485; no_music - 1542
    5.music.arff music - 16387; no_music - 1581
    6.music.arff music - 7932; no_music - 10028
    7.music.arff music - 16997; no_music - 963
    8.music.arff music - 12117; no_music - 5844
    9.music.arff music - 12975; no_music - 5032
    10.music.arff music - 6780; no_music - 11173
    11.music.arff music - 17341; no_music - 613
    12.music.arff music - 16440; no_music - 1754
    13.music.arff music - 15190; no_music - 3132
    14.music.arff music - 13941; no_music - 4382

-------------------> SPEECH <-------------------  
    
    1.speech.arff speech - 2140; no_speech - 15753
    2.speech.arff speech - 3176; no_speech - 14040
    3.speech.arff speech - 7968; no_speech - 9993
    4.speech.arff speech - 3595; no_speech - 14432
    5.speech.arff speech - 4537; no_speech - 13431
    6.speech.arff speech - 10250; no_speech - 7710
    7.speech.arff speech - 1069; no_speech - 16891
    8.speech.arff speech - 7837; no_speech - 10124
    9.speech.arff speech - 11598; no_speech - 6409
    10.speech.arff speech - 11110; no_speech - 6843
    11.speech.arff speech - 5335; no_speech - 12619
    12.speech.arff speech - 4104; no_speech - 14090
    13.speech.arff speech - 2861; no_speech - 15461
    14.speech.arff speech - 5051; no_speech - 13272  
"""


def print_nr_instances_per_class_per_file_by_data_type(
        root_dir=r'f:\SharedData\JKU\GIT\JKU-Computer-Science-Courses\ML-PatternClassification\train', data_type=None):
    if data_type is None:
        data_type = 'speech'

    for i in range(1, 15):
        file_path = '{}{}{}.{}.arff'.format(root_dir, os.sep, i, data_type)
        file_name = '{}.{}.arff'.format(i, data_type)
        data = arff.loadarff(file_path)
        df = pd.DataFrame(data[0])
        data_type_nr = df[df['class'] == str.encode(data_type)].shape[0]
        no_data_type_nr = df[df['class'] == str.encode('no_' + data_type)].shape[0]
        print("{} {} - {}; no_{} - {}".format(file_name, data_type, data_type_nr, data_type, no_data_type_nr))


music = [17431, 16877, 12393, 16485, 16387, 7932, 16997, 12117, 12975, 6780, 17341, 16440, 15190, 13941]

no_music = [462, 339, 5568, 1542, 1581, 10028, 963, 5844, 5032, 11173, 613, 1754, 3132, 4382]

speech = [2140, 3176, 7968, 3595, 4537, 10250, 1069, 7837, 11598, 11110, 5335, 4104, 2861, 5051]

no_speech = [15753, 14040, 9993, 14432, 13431, 7710, 16891, 10124, 6409, 6843, 12619, 14090, 15461, 13272]

# 0
train_music_60_percent = sum(music) * 60 / 100
train_no_music_60_percent = sum(no_music) * 60 / 100

# 1
test_no_music_20_percent = sum(no_music) * 20 / 100
test_music_20_percent = sum(music) * 20 / 100

# 2
validation_no_music_20_percent = sum(no_music) * 20 / 100
validation_music_20_percent = sum(music) * 20 / 100

# 0
train_speech_60_percent = sum(speech) * 60 / 100
train_no_speech_60_percent = sum(no_speech) * 60 / 100

# 1
test_no_speech_20_percent = sum(no_speech) * 20 / 100
test_speech_20_percent = sum(speech) * 20 / 100

# 2
validation_no_speech_20_percent = sum(no_speech) * 20 / 100
validation_speech_20_percent = sum(speech) * 20 / 100

min_diff = math.inf
config = []
top_ten_best_cfgs = [list(), list(), list(), list(), list(), list(), list(), list(), list(), list()]
a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# print(len(a))
# for i in range(len(a)):
#     for a[i] in range(3):
#         print(a)

total_nr = 0


def optim_speech(current_cfg):
    global train_speech_60_percent, train_no_speech_60_percent, test_speech_20_percent
    global test_no_speech_20_percent, validation_no_speech_20_percent, validation_speech_20_percent
    train_speech = 0
    train_no_speech = 0
    test_speech = 0
    test_no_speech = 0
    vali_speech = 0
    vali_no_speech = 0
    for i in range(len(current_cfg)):
        if current_cfg[i] == 0:
            train_speech += speech[i]
            train_no_speech += no_speech[i]
        elif current_cfg[i] == 1:
            test_speech += speech[i]
            test_no_speech += no_speech[i]
        else:
            vali_speech += speech[i]
            vali_no_speech += no_speech[i]
    ret_val = abs(train_speech - train_speech_60_percent) + \
              abs(train_no_speech - train_no_speech_60_percent) + \
              abs(test_speech - test_speech_20_percent) + \
              abs(test_no_speech - test_no_speech_20_percent) + \
              abs(vali_speech - validation_speech_20_percent) + \
              abs(vali_no_speech - validation_no_speech_20_percent)
    if 0 not in current_cfg or 1 not in current_cfg or 2 not in current_cfg:
        return math.inf
    else:
        return ret_val


def optim_music(a):
    global train_music_60_percent, train_no_music_60_percent, test_music_20_percent
    global test_no_music_20_percent, validation_no_music_20_percent, validation_music_20_percent
    train_music = 0
    train_no_music = 0
    test_music = 0
    test_no_music = 0
    vali_music = 0
    vali_no_music = 0
    for i in range(len(a)):
        if a[i] == 0:
            train_music += music[i]
            train_no_music += no_music[i]
        elif a[i] == 1:
            test_music += music[i]
            test_no_music += no_music[i]
        else:
            vali_music += music[i]
            vali_no_music += no_music[i]
    return abs(train_music - train_music_60_percent) + \
           abs(train_no_music - train_no_music_60_percent) + \
           abs(test_music - test_music_20_percent) + \
           abs(test_no_music - test_no_music_20_percent) + \
           abs(vali_music - validation_music_20_percent) + \
           abs(vali_no_music - validation_no_music_20_percent)


def generate_splits(a, i):
    global total_nr
    global min_diff, config
    global top_ten_best_cfgs
    if i >= len(a):
        return
    for a[i] in range(3):
        # print(a)
        yield a
        total_nr += 1
        if total_nr % 10000 == 0:
            print(total_nr)
        yield from generate_splits(a, i + 1)


def get_best_10_cfgs_for_music():
    global min_diff
    for cfg in generate_splits(a, 0):
        cat_de_optim = optim_music(cfg)
        if cat_de_optim < min_diff:
            min_diff = cat_de_optim
        top_ten_best_cfgs.pop(0)
        ne = copy.deepcopy(cfg)
        ne.append(min_diff)
        top_ten_best_cfgs.append(ne)
        config = cfg

    print(top_ten_best_cfgs)
    print("best:")
    print(config)
    print(min_diff)


def get_best_10_cfgs_for_speech():
    min_diff = math.inf
    for cfg in generate_splits(a, 0):
        # print(cfg)
        cat_de_optim = optim_speech(copy.deepcopy(cfg))
        if cat_de_optim < min_diff:
            min_diff = cat_de_optim
        top_ten_best_cfgs.pop(0)
        ne = copy.deepcopy(cfg)
        ne.append(copy.deepcopy(min_diff))
        top_ten_best_cfgs.append(ne)
        config = cfg

    print(top_ten_best_cfgs)
    print("best:")
    print(config)
    print(min_diff)


def music_no_music_split():
    best_music_cfgs = [[0, 0, 0, 0, 1, 0, 1, 2, 2, 1, 2, 0, 0, 0, 14246.80000000001],
                       [0, 0, 1, 0, 1, 0, 2, 1, 0, 2, 2, 0, 0, 0, 14154.80000000001],
                       [0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 0, 2, 13934.80000000001],
                       [0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 2, 2, 0, 1, 13842.80000000001],
                       [0, 1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 0, 0, 13210.80000000001],
                       [0, 1, 0, 0, 0, 0, 1, 2, 2, 1, 2, 0, 0, 0, 12742.80000000001],
                       [0, 1, 0, 0, 1, 1, 2, 0, 0, 2, 2, 0, 0, 0, 12662.80000000001],
                       [0, 1, 1, 0, 0, 0, 2, 1, 0, 2, 2, 0, 0, 0, 12650.80000000001],
                       [1, 1, 0, 0, 0, 0, 0, 2, 2, 1, 2, 0, 0, 0, 12608.80000000001],
                       [1, 1, 0, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 0, 12512.80000000001]]

    info_music_files = [
        "1.music.arff music - 17431; no_music - 462",
        "2.music.arff music - 16877; no_music - 339",
        "3.music.arff music - 12393; no_music - 5568",
        "4.music.arff music - 16485; no_music - 1542",
        "5.music.arff music - 16387; no_music - 1581",
        "6.music.arff music - 7932; no_music - 10028",
        "7.music.arff music - 16997; no_music - 963",
        "8.music.arff music - 12117; no_music - 5844",
        "9.music.arff music - 12975; no_music - 5032",
        "10.music.arff music - 6780; no_music - 11173",
        "11.music.arff music - 17341; no_music - 613",
        "12.music.arff music - 16440; no_music - 1754",
        "13.music.arff music - 15190; no_music - 3132",
        "14.music.arff music - 13941; no_music - 4382]"]

    print("For TRAINING")
    music_for_training = 0
    no_music_for_training = 0
    for i in range(len(best_music_cfgs[-1]) - 1):
        if best_music_cfgs[-1][i] == 0:
            print(info_music_files[i])
            music_for_training += music[i]
            no_music_for_training += no_music[i]
    print("Total number of MUSIC instances for TRAINING = {}; {}% of dataset".format(music_for_training, (
            music_for_training / sum(music)) * 100))
    print("Total number of NO_MUSIC instances for TRAINING = {}; {}% of dataset".format(no_music_for_training, (
            no_music_for_training / sum(no_music)) * 100))

    print("\n==================================================\n")
    music_for_test = 0
    no_music_for_test = 0
    print("For TEST")
    for i in range(len(best_music_cfgs[-1]) - 1):
        if best_music_cfgs[-1][i] == 1:
            print(info_music_files[i])
            music_for_test += music[i]
            no_music_for_test += no_music[i]

    print("Total number of MUSIC instances for TEST = {}; {}% of dataset".format(music_for_test, (
            music_for_test / sum(music)) * 100))
    print("Total number of NO_MUSIC instances for TEST = {}; {}% of dataset".format(no_music_for_test, (
            no_music_for_test / sum(no_music)) * 100))

    print("\n==================================================\n")
    music_for_validation = 0
    no_music_for_validation = 0
    print("For VALIDATION")
    for i in range(len(best_music_cfgs[-1]) - 1):
        if best_music_cfgs[-1][i] == 2:
            print(info_music_files[i])
            music_for_validation += music[i]
            no_music_for_validation += no_music[i]

    print("Total number of MUSIC instances for VALIDATION = {}; {}% of dataset".format(music_for_validation, (
            music_for_validation / sum(music)) * 100))
    print("Total number of NO_MUSIC instances for VALIDATION = {}; {}% of dataset".format(no_music_for_validation, (
            no_music_for_validation / sum(no_music)) * 100))


def speech_no_speech_split():
    best_speech_cfgs = [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 12512.800000000003],
                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 12512.800000000003],
                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 12512.800000000003],
                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 12512.800000000003],
                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 12512.800000000003],
                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 12512.800000000003],
                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 12512.800000000003],
                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 12512.800000000003],
                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 12512.800000000003],
                        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 12512.800000000003]]

    info_speech_files = [
        "1.speech.arff speech - 17431; no_speech - 462",
        "2.speech.arff speech - 16877; no_speech - 339",
        "3.speech.arff speech - 12393; no_speech - 5568",
        "4.speech.arff speech - 16485; no_speech - 1542",
        "5.speech.arff speech - 16387; no_speech - 1581",
        "6.speech.arff speech - 7932; no_speech - 10028",
        "7.speech.arff speech - 16997; no_speech - 963",
        "8.speech.arff speech - 12117; no_speech - 5844",
        "9.speech.arff speech - 12975; no_speech - 5032",
        "10.speech.arff speech - 6780; no_speech - 11173",
        "11.speech.arff speech - 17341; no_speech - 613",
        "12.speech.arff speech - 16440; no_speech - 1754",
        "13.speech.arff speech - 15190; no_speech - 3132",
        "14.speech.arff speech - 13941; no_speech - 4382]"]

    print("For TRAINING")
    speech_for_training = 0
    no_speech_for_training = 0
    for i in range(len(best_speech_cfgs[-1]) - 1):
        if best_speech_cfgs[-1][i] == 0:
            print(info_speech_files[i])
            speech_for_training += speech[i]
            no_speech_for_training += no_speech[i]
    print("Total number of MUSIC instances for TRAINING = {}; {}% of dataset".format(speech_for_training, (
            speech_for_training / sum(speech)) * 100))
    print("Total number of NO_MUSIC instances for TRAINING = {}; {}% of dataset".format(no_speech_for_training, (
            no_speech_for_training / sum(no_speech)) * 100))

    print("\n==================================================\n")
    speech_for_test = 0
    no_speech_for_test = 0
    print("For TEST")
    for i in range(len(best_speech_cfgs[-1]) - 1):
        if best_speech_cfgs[-1][i] == 1:
            print(info_speech_files[i])
            speech_for_test += speech[i]
            no_speech_for_test += no_speech[i]

    print("Total number of MUSIC instances for TEST = {}; {}% of dataset".format(speech_for_test, (
            speech_for_test / sum(speech)) * 100))
    print("Total number of NO_MUSIC instances for TEST = {}; {}% of dataset".format(no_speech_for_test, (
            no_speech_for_test / sum(no_speech)) * 100))

    print("\n==================================================\n")
    speech_for_validation = 0
    no_speech_for_validation = 0
    print("For VALIDATION")
    for i in range(len(best_speech_cfgs[-1]) - 1):
        if best_speech_cfgs[-1][i] == 2:
            print(info_speech_files[i])
            speech_for_validation += speech[i]
            no_speech_for_validation += no_speech[i]

    print("Total number of MUSIC instances for VALIDATION = {}; {}% of dataset".format(speech_for_validation, (
            speech_for_validation / sum(speech)) * 100))
    print("Total number of NO_MUSIC instances for VALIDATION = {}; {}% of dataset".format(no_speech_for_validation, (
            no_speech_for_validation / sum(no_music)) * 100))


get_best_10_cfgs_for_music()

