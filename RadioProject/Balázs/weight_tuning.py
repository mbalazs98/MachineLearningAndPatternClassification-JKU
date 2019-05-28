import matplotlib.pyplot as plt
from io_operations import DataLoader
from classifier_orchestrator import ClassifierOrchestrator

#This function plots how the accuracies changes by adjusting the weights from 0 to 1. It excepts an array of standard weights, and it plots multiple times, always changing the weight of one prediction.
def plot_vote_weight_acc(data_type,preds,std_weights):
	y=load_test_data(data_type)['class'].tolist()
	total=len(preds)
	if data_type=='music':
		t=b'music'
		f=b'no_music'
		ind=0
	else:
		t=b'speech'
		f=b'no_speech'
		ind=1
	col=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
	for i in range(len(std_weights)):
		acc_arr=[]
		for j in range(0,1,0.1):
			tmp_weights=std_weights
			tmp_weights[i]=j
			acc=0
			for k in range(len(preds)):
				fin=get_final_prediction_for_one_instance_weighed(preds[k],tmp_weights)[ind]
				if (fin==1 and y[k]==t) or (fin==0 and y[k]==f):
					acc=acc+1
			acc_arr.append(acc/total)
		plt.plot(range(0,1,0.1),acc_arr,col[i])
	plt.show()
