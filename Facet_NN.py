import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, Dropout, MaxPooling1D, Normalization
from tensorflow.keras.optimizers import SGD, Adam, Adagrad
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback

# GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Custom Stop
class myCallback(Callback):    
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] >= 0.05:
               self.model.stop_training = True

# Possible architecture
def Build_Model(hp):
    model = Sequential()
    model.add(Input(shape=(14,)))
    for i in range(hp.Int('layers', 1, 3)):
        model.add(Dense(
            units=hp.Choice('units_' + str(i), [3, 9, 27]),
            activation=hp.Choice('act_' + str(i), ['relu', 'sigmoid'])))
    model.add(Dense(1, hp.Choice('activation', ["linear", 'relu', 'sigmoid'])))
    learning_rate = hp.Float("lr", min_value=0.00001, max_value=0.01)
    model.compile(loss='mean_squared_error', optimizer=Adagrad(learning_rate=learning_rate))
    return model

# Search for each fold
for k in range(12):
    # Entire Fold
    train, test = pd.read_csv('Data/Train/data_train_{}.csv'.format(k)), pd.read_csv('Data/Test/data_test_{}.csv'.format(k))
    complete = pd.concat([train, test]).reset_index().drop(['index'], axis=1)
    features = np.array(complete[['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']], dtype=np.float64)
    accuracy = np.array(complete['19'], dtype=np.float64).reshape(-1, 1)

    # Training Data
    train_features = np.array(train[['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']], dtype=np.float64)
    train_accuracy = np.array(train['19'], dtype=np.float64).reshape(-1, 1)

    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(train_features, train_accuracy, test_size=0.4, random_state=0)

    # Sample_weights
    sample_weights_train = class_weight.compute_sample_weight(class_weight='balanced', y=y_train).reshape(-1,1)
    sample_weights_test = class_weight.compute_sample_weight(class_weight='balanced', y=y_test).reshape(-1,1)

    # Model
    tuner = kt.RandomSearch(
        Build_Model,
        objective='val_mse',
        max_trials=20,
        directory='Facet_NN_Tuner',
        project_name='Facet_NN_Fold_{}'.format(k))

    # Early Stop
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-3)
    my_loss_callback = myCallback()

    # Search
    tuner.search(
        x_train, 
        y_train, 
        epochs=500, 
        validation_data=(x_test, y_test, sample_weights_test), 
        sample_weight=sample_weights_train, 
        callbacks=[early_stopping, my_loss_callback],
        verbose=1)

    # Get Model
    best_model = tuner.get_best_models()[0]

    # Create .out
    param_bench = complete[['0','1']]
    param_bench['out'] = pd.DataFrame(best_model.predict(features))

    # Write .out
    final_accuracy_file = []
    for i in tqdm(range(len(param_bench))):
        final_accuracy_file.append(str(param_bench['0'][i]) + '/' + str(param_bench['1'][i]) + '\t' + str(param_bench['out'][i]))

    with open('Estimator/Facet_NN_Estimator/Facet_NN_Fold_{}.out'.format(k), 'w') as txt_file:
        for line in final_accuracy_file:
            txt_file.write("".join(line) + "\n")

