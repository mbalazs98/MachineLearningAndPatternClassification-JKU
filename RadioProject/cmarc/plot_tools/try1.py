from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt

datal = []
dfl = []
for i in range(1, 2):
    data = arff.loadarff(
        r'f:\SharedData\JKU\GIT\JKU-Computer-Science-Courses\ML-PatternClassification\train\{}.music.arff'.format(i))
    df = pd.DataFrame(data[0])
    datal.append(data)
    dfl.append(df)
# data = pd.concat(datal)
df = pd.concat(dfl)
with open("corr.txt", "w") as g:
    g.write(str(df.corr(method='pearson')))
# df['f000704'].plot(kind='hist', bins=50, figsize=(12, 6))
# df['f000705'].plot(kind='hist', bins=50, figsize=(12, 6))

music = df[df['class'] == b'music']
no_music = df[df['class'] == b'no_music']

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# ax1 = music['f000704'].plot(kind='hist', bins=50, figsize=(12, 6))
# plt.show()
# ax2 = no_music['f000704'].plot(kind='hist', bins=50, figsize=(12, 6))
# plt.show()
ax3 = music['f000705'].plot(kind='hist', bins=50, figsize=(12, 6))
# plt.show()
ax4 = no_music['f000705'].plot(kind='hist', bins=50, figsize=(12, 6))


# fig.add_subplot(ax1, label="704 music")
# fig.add_subplot(ax2, label="704 no music")
ax3.set_title("705 music")
fig.add_subplot(ax3)
ax4.set_title("705 no music")
fig.add_subplot(ax4)

plt.show()

# df.plot(kind='scatter', x='f000704', y='f000705')
# ax = music.plot(kind='scatter', x='f000704', y='f000705', color = "green", label='music')
# no_music.plot(kind='scatter', x='f000704', y='f000705', color = "red", label='no music', ax = ax)
# w

print(df.head(10))
# print(music.head(10))
# print(no_music.head(10))
b = input()
