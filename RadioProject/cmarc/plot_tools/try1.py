from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt

datal = []
dfl = []
for i in range(1, 2):
    data = arff.loadarff(r'f:\SharedData\JKU\GIT\JKU-Computer-Science-Courses\ML-PatternClassification\train\{}.music.arff'.format(i))
    df = pd.DataFrame(data[0])
    datal.append(data)
    dfl.append(df)
# data = pd.concat(datal)
df = pd.concat(dfl)
print(df.corr(method='pearson'))
# df['f000704'].plot(kind='hist', bins=50, figsize=(12, 6))
df['f000705'].plot(kind='hist', bins=50, figsize=(12, 6))

# music = df[df['class'] == b'music']
# no_music = df[df['class'] == b'no_music']
# df.plot(kind='scatter', x='f000704', y='f000705')
# ax = music.plot(kind='scatter', x='f000704', y='f000705', color = "green", label='music')
# no_music.plot(kind='scatter', x='f000704', y='f000705', color = "red", label='no music', ax = ax)
plt.show()  # w

print(df.head(10))
# print(music.head(10))
# print(no_music.head(10))
b = input()
