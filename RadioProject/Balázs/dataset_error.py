from scipy.io import arff

trainInd = [3,4,5,6,9,12,13,14]

train=0
all=0
for i in range(1,14):
	data,meta = arff.loadarff(r'D:\mlpc\train\{}.music.arff'.format(i))

	music=data['class']
	
	data,meta = arff.loadarff(r'D:\mlpc\train\{}.speech.arff'.format(i))

	speech=data['class']

	n=music.size
	for j in range(0,n):
		if music[j]==b'no_music' and speech[j]==b'no_speech':
			if i in trainInd:
				train=train+1
			all=all+1
	print(all)
print(train)#1900 instances in trainng data when it is classified as no_music and no_speech at the same time
print(all)#2137 instances in the whole dataset when it is classified as no_music and no_speech at the same time