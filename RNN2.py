
'''
This Python file loads the data from the respective pickle files
and carries out different RNN models to predict falls from ADL's.
'''

import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.callbacks import EarlyStopping

# Loading data to go through neural networks
X1 = pickle.load(open("SamplesBF_RNN.pickle", "rb"))
Y1 = pickle.load(open("LabelsBF_RNN.pickle", "rb"))
X2 = pickle.load(open("SamplesAF_RNN.pickle", "rb"))
Y2 = pickle.load(open("LabelsAF_RNN.pickle", "rb"))

print('Before Filtering: Shape of X1:', X1.shape, 'Shape of Y1:', Y1.shape)
print('After Filtering: Shape of X2:', X2.shape, 'Shape of Y2:', Y2.shape)

# Separating Train & Test Datasets
#split_size = int(X.shape[0]*0.7)
#X1_train, X1_test = X[:split_size], X[split_size:]
#Y1_train, Y1_test = Y[:split_size], Y[split_size:]

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.3, random_state = 0)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.3, random_state = 0)

print('Before Filtering: Shape of X1_train:', X1_train.shape, 'Shape of Y1_train:', Y1_train.shape)
print('After Filtering: Shape of X2_train:', X2_train.shape, 'Shape of Y2_train:', Y2_train.shape)

unique_values1 = np.unique(Y1_test)
print("Unique values in Y1_test:", unique_values1)

unique_values2 = np.unique(Y2_test)
print("Unique values in Y2_test:", unique_values2)

# Building LSTM-RNN Model
modelBF_RNN = tf.keras.models.Sequential\
([
     tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(X1_train.shape[1], X1_train.shape[2])),    # Takes number of timesteps & features as input
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.LSTM(64, return_sequences=True),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.LSTM(32, return_sequences=True),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dense(16),
     tf.keras.layers.Dropout(0.1),
     tf.keras.layers.Dense(2, activation='sigmoid')
])
modelBF_RNN.summary()     # Displays parameters within model
modelBF_RNN.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])


modelAF_RNN = tf.keras.models.Sequential\
([
     tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(X1_train.shape[1], X1_train.shape[2])),    # Takes number of timesteps & features as input
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.LSTM(64, return_sequences=True),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.LSTM(32, return_sequences=True),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dense(16),
     tf.keras.layers.Dropout(0.1),
     tf.keras.layers.Dense(2, activation='sigmoid')
])
modelAF_RNN.summary()     # Displays parameters within model
modelAF_RNN.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=15, 
    min_delta=0.001, 
    mode='max'
)

# Training model
print("Training model w/ data before filtering...")
historyBF_RNN = modelBF_RNN.fit(X1_train, Y1_train, validation_data=(X1_test, Y1_test), batch_size=40, epochs=1, shuffle = False, verbose=1,callbacks=[early_stopping])
#y1_pred_BF = modelBF_RNN.predict(X1_test, verbose=0)
y1_pred_BF = (modelBF_RNN.predict(X1_test) > 0.5).astype("int32")
Y1_test_BF = Y1_test.astype("int32")
print(Y1_test_BF.dtype, y1_pred_BF.dtype)
print(Y1_test_BF.shape, y1_pred_BF.shape)

unique_values3 = np.unique(y1_pred_BF)
print("Unique values in y1_pred_BF:", unique_values3)

unique_values4 = np.unique(Y1_test_BF)
print("Unique values in Y1_test_BF:", unique_values4)
'''
y1_pred_class_BF = np.argmax(y1_pred_BF,axis=1)
Y1_test_labels = np.argmax(Y1_test, axis=1)
print(Y1_test_labels.dtype, y1_pred_class_BF.dtype)
print(Y1_test_labels.shape, y1_pred_class_BF.shape)


if Y1_test.ndim == 2 and Y1_test.shape[1] > 1:
    Y1_test_labels = np.argmax(Y1_test, axis=1)

'''

print("Training model w/ data after filtering...")
historyAF_RNN = modelAF_RNN.fit(X2_train, Y2_train, validation_data=(X2_test, Y2_test), batch_size=40, epochs=1, shuffle = False, verbose=1,callbacks=[early_stopping])
#y2_pred_AF = modelAF_RNN.predict(X2_test, verbose=0)
y2_pred_AF = (modelAF_RNN.predict(X2_test) > 0.5).astype("int32")
Y2_test_AF = Y2_test.astype("int32")

print(Y2_test_AF.dtype, y2_pred_AF.dtype)
print(Y2_test_AF.shape, y2_pred_AF.shape)

unique_values5 = np.unique(y2_pred_AF)
print("Unique values in y2_pred_AF:", unique_values5)

unique_values6 = np.unique(Y2_test_AF)
print("Unique values in Y2_test_AF:", unique_values6)
'''
if Y2_test.ndim == 2 and Y2_test.shape[1] > 1:
    Y2_test_labels = np.argmax(Y2_test, axis=1)
y2_pred_class_AF = np.argmax(y2_pred_AF,axis=1)
Y2_test_labels = np.argmax(Y2_test, axis=1)
print(Y2_test_labels.dtype, y2_pred_class_AF.dtype)
print(Y2_test_labels.shape, y2_pred_class_AF.shape)
'''

# Prediction Stage
print("Evaluating models...")
loss1, acc1 = modelBF_RNN.evaluate(X1_test, Y1_test, verbose=1)
loss2, acc2 = modelAF_RNN.evaluate(X2_test, Y2_test, verbose=1)
print("Evaluated Accuracy")
print("------------------")
print("Before Filter: {:4.4f}%" .format(100*acc1))
print("After Filter: {:4.4f}%" .format(100*acc2))
print("Evaluated Loss")
print("------------------")
print("Before Filter: {:4.4f}%" .format(100*loss1))
print("After Filter: {:4.4f}%" .format(100*loss2))

#Classification Report

from sklearn.metrics import classification_report
print("@@@@@@ Classification Report - Before Filtering @@@@@")
print(classification_report(Y1_test_BF, y1_pred_BF))
print("@@@@@@ Classification Report - After Filtering @@@@@")
print(classification_report(Y2_test_AF, y2_pred_AF))


cf_mat_bf = confusion_matrix(Y1_test_BF, y1_pred_BF)
print('Confusion Matrix - Before Filter')
print(cf_mat_bf)
cf_mat_af = confusion_matrix(Y2_test_AF, y2_pred_AF)
print('Confusion Matrix - After Filter')
print(cf_mat_af)



# Saving Stage
print("Saving history of model without filtering...")
pickle_out = open("HistoryBF_RNN.pickle", "wb")
pickle.dump(historyBF_RNN.history, pickle_out)
pickle_out.close()

print("Saving history of model with filtering...")
pickle_out = open("HistoryAF_RNN.pickle", "wb")
pickle.dump(historyAF_RNN.history, pickle_out)
pickle_out.close()
