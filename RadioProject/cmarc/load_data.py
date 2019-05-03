import time

from scipy.io import arff
import numpy as np
import arff


def try_1():
    datal = []
    dfl = []
    all = None
    all_x = None
    all_y = None
    for i in range(1, 4):
        data = arff.loadarff(
            r'af:\SharedData\JKU\GIT\JKU-Computer-Science-Courses\ML-PatternClassification\train\{}.music.arff'.format(
                i))
        a = data[0]
        if all is None:
            all = np.array(a)
        else:
            all += np.array(a)

    print(all.shape)
    print(all)
    for item in all:
        if all_x is None:
            all_x = a[:-1]
        else:
            all_x += a[:-1]
        if all_y is None:
            all_y = a[-1:]
        else:
            all_y += a[-1:]

    print(all_x.shape)
    print(all_y.shape)
    print(all_x)
    print(all_y)


def load_music_i_from_file(i):
    data = arff.load(
        open(
            r'f:\SharedData\JKU\GIT\JKU-Computer-Science-Courses\ML-PatternClassification\train\{}.music.arff'.format(
                i)))
    return data


def load_first_2():
    x = np.load("x1-2.txt.npy")
    y = np.load("y1-2.txt.npy")
    return x, y


def try_2():
    datas = []
    start = time.time()

    for i in range(1, 15):
        data = arff.load(
            open(
                r'f:\SharedData\JKU\GIT\JKU-Computer-Science-Courses\ML-PatternClassification\train\{}.music.arff'.format(
                    i)))
        a = np.asarray(data["data"])
        np.save(
            r'f:\SharedData\JKU\GIT\JKU-Computer-Science-Courses\ML-PatternClassification\train\{}.music.arff.npy'.format(
                i), a)
    #     np.save("y1-14", y)
    #     datas.append(a)
    # all_data = np.concatenate(datas)
    # x = all_data[:, :-1]
    # y = all_data[:, -1]
    #
    # # x = np.load("x1-2.txt.npy")
    # # y = np.load("y1-2.txt.npy")
    # print(x.shape)
    # print(y.shape)
    # np.save("x1-14", x)
    # np.save("y1-14", y)
    # print("Total load time = {} sec".format(time.time()-start))
