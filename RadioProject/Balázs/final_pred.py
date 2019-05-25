def get_final_prediction(preds, weights):
	music_pred=[]
	speech_pred=[]
	for i in preds:
		music=0
		no_music=0
		for j in preds[i].music:
			if preds[i].music[j]==1:
				music=music+weights.music[j]
			else:
				no_music=no_music+weights.music[j]
		speech=0
		no_speech=0
		for j in preds[i].speech:
			if preds[i].speech[j]==1:
				speech=speech+weights.speech[j]
			else:
				no_speech=no_speech+1
		if music>no_music:
			music_pred.add(1)
		else:
			music_pred.add(0)
		if speech>no_speech:
			speech_pred.add(1)
		else:
			speech_pred.add(0)
	return music_pred,speech_pred
		
		
	