from tensorflow.keras import layers,models
from tensorflow.keras import callbacks
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.utils import class_weight
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="2"

def identity_block(input_tensor,units):
	x = layers.Dense(units)(input_tensor)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)

	x = layers.add([x, input_tensor])
	x = layers.Activation('relu')(x)

	return x

def dens_block(input_tensor,units):
	x = layers.Dense(units)(input_tensor)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units)(x)
	x = layers.BatchNormalization()(x)

	shortcut = layers.Dense(units)(input_tensor)
	shortcut = layers.BatchNormalization()(shortcut)

	x = layers.add([x, shortcut])
	x = layers.Activation('relu')(x)
	return x


def ResNet50Regression():
	Res_input = layers.Input(shape=(14,))
	width = 9

	x = dens_block(Res_input,width)
	x = identity_block(x,width)
	x = identity_block(x,width)

	x = dens_block(x,width)
	x = identity_block(x,width)
	x = identity_block(x,width)
	
	x = layers.BatchNormalization()(x)
	x = layers.Dense(1, activation='relu')(x)
	# x = layers.Dense(1, activation='linear')(x) v1
	model = models.Model(inputs=Res_input, outputs=x)

	return model


k = 2
train, test = pd.read_csv('Data/Train/data_train_{}.csv'.format(k)), pd.read_csv('Data/Test/data_test_{}.csv'.format(k))
complete = pd.concat([train, test]).reset_index().drop(['index'], axis=1)
features = np.array(complete[['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']], dtype=np.float64)
accuracy = np.array(complete['19'], dtype=np.float64).reshape(-1, 1)

# Training Data
train_features = np.array(train[['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']], dtype=np.float64)
train_accuracy = np.array(train['19'], dtype=np.float64).reshape(-1, 1)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(train_features, train_accuracy, test_size=0.4, random_state=0)

# Sample_weights
sample_weights_train = class_weight.compute_sample_weight(class_weight='balanced', y=y_train).reshape(-1,1)
sample_weights_test = class_weight.compute_sample_weight(class_weight='balanced', y=y_test).reshape(-1,1)

model = ResNet50Regression()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.summary()

history = model.fit(X_train, y_train, epochs=70, verbose=1, callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2, mode='auto')], validation_data=(X_test, y_test, sample_weights_test), 
        sample_weight=sample_weights_train)

yhat = model.predict(features)
print(mean_squared_error(yhat, accuracy))
print(r2_score(yhat, accuracy))

# Create .out
param_bench = complete[['0','1']]
param_bench['out'] = pd.DataFrame(yhat)

# Write .out
final_accuracy_file = []
for i in tqdm(range(len(param_bench))):
    final_accuracy_file.append(str(param_bench['0'][i]) + '/' + str(param_bench['1'][i]) + '\t' + str(param_bench['out'][i]))

with open('Estimator/Facet_ResNet_Estimator/Facet_ResNet_Fold_{}.out'.format(k), 'w') as txt_file:
    for line in final_accuracy_file:
        txt_file.write("".join(line) + "\n")



