from keras.utils import to_categorical
from keras.layers import Merge
from keras import regularizers

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

import tensorflow as tf

import numpy as np
import pandas as pd

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []

	def on_epoch_end(self, epoch, logs={}):
#		print("")
		print(self.model)
	
		val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
#self.model.validation_data[0]

#val_predict = (np.asarray(self.model.predict(self.model.validation_data))).round()
#		print("predict")
#		print(val_predict)
		val_targ = self.model.validation_data[1]
		_val_f1 = f1_score(val_targ, val_predict)
		_val_recall = recall_score(val_targ, val_predict)
		_val_precision = precision_score(val_targ, val_predict)
		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		print ("— val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
	
		return


def create_model_for_merge_3(optimizer_, X1, y1, X2, y2, X3, y3, rate1, rate2):
#with tf.device('/cpu:0'):
	   # model generation
	model1 = Sequential()
	model1.add(Dense(512, input_dim=X1.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.001)))
	model1.add(Dropout(rate1))
	model1.add(Dense(256, activation='relu'))
	model1.add(Dropout(rate1))
	model1.add(Dense(128, activation='relu'))
	model1.add(Dropout(rate2))
	
	# model generation
	model2 = Sequential()
	model2.add(Dense(256, input_dim=X2.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.001)))
	model2.add(Dropout(rate1))
	model2.add(Dense(128, activation='relu'))
	model2.add(Dropout(rate1))
	model2.add(Dense(128, activation='relu'))
	model2.add(Dropout(rate2))
	
	# model generation
	model3 = Sequential()
	model3.add(Dense(256, input_dim=X3.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.001)))
	model3.add(Dropout(rate1))
	model3.add(Dense(128, activation='relu'))
	model3.add(Dropout(rate1))
	model3.add(Dense(128, activation='relu'))
	model3.add(Dropout(rate2))
	
	# model merge
	model_merged = keras.layers.Add()([model1, model2, model3])
	#model_merged.add(Merge([model1, model2, model3], mode='concat'))
	#model_merged.add(Dense(256, activation='relu'))
	model_merged.add(Dropout(rate2))
	model_merged.add(Dense(y1.shape[1]))
	model_merged.add(Activation('softmax'))

#	parallel_model = multi_gpu_model(model_merged, gpus=2)
	model_merged.compile(loss='categorical_crossentropy',
			#optimizer='adam',
			optimizer=optimizer_,
			metrics=['accuracy'])

	return model_merged



X1 = pd.read_csv("VGG/joined/men_feminine_train_Image_df.csv", encoding='utf-8')
picname = X1['Picture_name'].tolist()
y_tmp = pd.read_csv("y/Y_Raw_2.csv", encoding='utf-8').drop(["Unnamed: 0"], axis=1)
y1 = y_tmp[y_tmp['Picture_name'].isin(picname)][['Feminine']]

y1 = to_categorical(y1)

X2_tmp = pd.read_csv("non_VGG/2_TEXT/Text_df.csv", encoding='utf-8').drop(["Unnamed: 0"], axis=1)
X2 = X2_tmp[X2_tmp['Picture_name'].isin(picname)]
X3_tmp = pd.read_csv("non_VGG/3_Activity/10_Activity.csv", encoding='utf-8').drop(["Unnamed: 0"], axis=1)
X3 = X3_tmp[X3_tmp['Picture_name'].isin(picname)]

rate1 = 0.5
rate2 = 0.5


X1 = X1.drop(['Picture_name'], axis=1)
X2 = X2.drop(['Picture_name'], axis=1)
X3 = X3.drop(['Picture_name'], axis=1)


for rate1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
	for rate2 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
		NN_merged = create_model_for_merge_3('adam', X1, y1, X2, y1, X3, y1, rate1, rate2)


		metrics = Metrics()
		model_merged = NN_merged.fit([X1, X2, X3], y1, 
				validation_split=0.2, 
				epochs=20, 
				batch_size=128, 
				verbose=1, 
				shuffle=True,)
#callbacks=[metrics]) 


#metrics = Metrics()
#print("F1: ", np.max(metrics.f1s))
#print("precision: ", np.max(metrics.precision))
#print("recall: ", np.max(metrics.recall))
#print("auc: ", np.max(metrics.auc))


