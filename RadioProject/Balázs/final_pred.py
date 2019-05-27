import pandas as pd
import matplotlib.pyplot as plt

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
    return load_data_from_list_of_files(data_type, TEST_FILES_NO)
	
def load_validation_data(data_type):
    return load_data_from_list_of_files(data_type, VALIDATION_FILES_NO)

def load_data_from_specific_file_no(
    root_dir=r'f:\SharedData\JKU\GIT\JKU-Computer-Science-Courses\ML-PatternClassification\train',data_type='music', file_no=1):
    data = arff.loadarff('{}{}{}.{}.arff'.format(root_dir, os.sep, file_no, data_type))
    df = pd.DataFrame(data[0])
    return df

#This function shows that by adjusting the weights, how much do the classifiers agree. Maximize this.
def try_weigths(preds, weights):
	final=[]
	t_all=0
	f_all=0
	for i in preds:
		t=0
		f=0
		for j in preds[i]:
			if preds[i][j]==1:
				t=t+weights[j]
				t_all=t_all+weights[j]
			else:
				f=f+weights[j]
				f_all=f_all+weights[j]
		final.add(abs(t-f))
	t_all=abs(t_all-f_all)/len(final)
	return final,t_all

#This function returns a prediction for a single instance of prediction, either music or speech
def get_final_prediction(pred,weights):
	t=0
	f=0
	for i in preds:
		if pred[i]==1:
			t=t+weights[i]
		else:
			f=f+weights[i]
	if t>f:
		return 1
	return 0
		
#This function returns the final predictions for all music and speech predictions
def get_full_final_prediction(preds, weights):
	music_pred=[]
	speech_pred=[]
	for i in preds:
		music_pred.add(get_final_prediction(preds[i].music,weights.music))
		speech_pred.add(get_final_prediction(preds[i].speech,weights.speech))
	return music_pred,speech_pred

#This function plots how the accuracies changes by adjusting the weights from 0 to 1. It excepts an array of standard weights, and it plots multiple times, always changing the weight of one prediction.
def plot_vote_weight_acc(data_type,preds,std_weights):
	y=load_test_data(data_type)['class'].tolist()
	total=len(preds)
	if data_type=='music':
		t=b'music'
		f=b'no_music'
	else:
		t=b'speech'
		f=b'no_speech'
	col=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
	for i in range(len(std_weights)):
		acc_arr=[]
		for j in range(0,1,0.1):
			tmp_weights=std_weights
			tmp_weights[i]=j
			acc=0
			for k in range(len(preds)):
				fin=get_final_prediction(preds[k],tmp_weights)
				if (fin==1 and y[k]==t) or (fin==0 and y[k]==f):
					acc=acc+1
			acc_arr.append(acc/total)
		plt.plot(range(0,1,0.1),acc_arr,col[i])
	plt.show()
			