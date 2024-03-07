
'''
This Python file loads the data from the respective pickle files
and carries out different RNN models to predict falls from ADL's.
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

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

# Building recurrent neural network model
modelBF_1DCNN = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X1_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
modelBF_1DCNN.summary()     # Displays parameters within model
modelBF_1DCNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

modelAF_1DCNN = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X2_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
modelAF_1DCNN.summary()     # Displays parameters within model
modelAF_1DCNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=15, 
    min_delta=0.001, 
    mode='max'
)

# Reshape for Conv1D
X1_train = X1_train.reshape((X1_train.shape[0], X1_train.shape[1], 1))
X1_test = X1_test.reshape((X1_test.shape[0], X1_test.shape[1], 1))
X2_train = X2_train.reshape((X2_train.shape[0], X2_train.shape[1], 1))
X2_test = X2_test.reshape((X2_test.shape[0], X2_test.shape[1], 1))

if Y1_train.ndim == 2:  # Assuming Y1_train is your label array
    Y1_train = to_categorical(Y1_train)
    print(Y1_train.shape)
if Y1_test.ndim == 2:  # Assuming Y1_test is your label array
    Y1_test = to_categorical(Y1_test)

# Training model
print("Training model w/ data before filtering...")
historyBF_1DCNN = modelBF_1DCNN.fit(X1_train, Y1_train, validation_data=(X1_test, Y1_test), batch_size=40, epochs=50, shuffle = False, verbose=1,callbacks=[early_stopping])
#history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

y1_pred_BF = modelBF_1DCNN.predict(X1_test, verbose=0)
#y1_pred_class_BF = modelBF_RNN.predict_classes(X1_test, verbose=0)
y1_pred_class_BF = np.argmax(y1_pred_BF,axis=1)

if Y2_train.ndim == 2:  # Assuming Y2_train is your label array
    Y2_train = to_categorical(Y2_train)
    print(Y2_train.shape)
if Y2_test.ndim == 2:  # Assuming Y2_test is your label array
    Y2_test = to_categorical(Y2_test)

print("Training model w/ data after filtering...")
historyAF_1DCNN = modelAF_1DCNN.fit(X2_train, Y2_train, validation_data=(X2_test, Y2_test), batch_size=40, epochs=50, shuffle = False, verbose=1,callbacks=[early_stopping])
y2_pred_AF = modelAF_1DCNN.predict(X2_test, verbose=0)
y2_pred_AF.to_csv('y2_pred.csv', index=False)       
y2_pred_class_AF = np.argmax(y2_pred_AF,axis=1)
y2_pred_class_AF.to_csv('y2_pred_AF.csv', index=False)  
# reduce to 1d array
y1_pred_BF = y1_pred_BF[:, 0]
y1_pred_class_BF = y1_pred_class_BF[:, 0]


y2_pred_AF = y2_pred_AF[:, 0]
y2_pred_class_AF = y2_pred_class_AF[:, 0]

# Prediction Stage
print("Evaluating models...")
loss1, acc1 = modelBF_1DCNN.evaluate(X1_test, Y1_test, verbose=1)
loss2, acc2 = modelAF_1DCNN.evaluate(X2_test, Y2_test, verbose=1)
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
print(classification_report(Y1_test, y1_pred_class_BF))
print("@@@@@@ Classification Report - After Filtering @@@@@")
print(classification_report(Y2_test, y2_pred_class_AF))

'''
cf_mat_bf = confusion_matrix(Y1_train, y1_pred_BF)
print('Confusion Matrix - Before Filter')
print(cf_mat_bf)
cf_mat_af = confusion_matrix(Y2_train, y2_pred_AF)
print('Confusion Matrix - Before Filter')
print(cf_mat_af)
'''
'''
#Classification metrics - Before Filtering
print ("Classification metrics - Before Filtering:")
# accuracy: (tp + tn) / (p + n)
accuracy_bf = accuracy_score(Y1_test, y1_pred_class_BF)
print('Accuracy: %f' % accuracy_bf)
# precision tp / (tp + fp)
precision_bf = precision_score(Y1_test, y1_pred_class_BF)
print('Precision: %f' % precision_bf)
# recall: tp / (tp + fn)
recall_bf = recall_score(Y1_test, y1_pred_class_BF)
print('Recall: %f' % recall_bf)
# f1: 2 tp / (2 tp + fp + fn)
f1_bf = f1_score(Y1_test, y1_pred_class_BF)
print('F1 score: %f' % f1_bf)
# kappa
kappa_bf = cohen_kappa_score(Y1_test, y1_pred_class_BF)
print('Cohens kappa: %f' % kappa_bf)
# ROC AUC
auc_bf = roc_auc_score(Y1_test, y1_pred_BF)
print('ROC AUC: %f' % auc_bf)
'''
# confusion matrix
matrix_bf = confusion_matrix(Y1_test, y1_pred_class_BF,normalize='pred')
print(matrix_bf)
'''
#Classification metrics - After Filtering
print ("Classification metrics - After Filtering:")
# accuracy: (tp + tn) / (p + n)
accuracy_af = accuracy_score(Y2_test, y2_pred_class_AF)
print('Accuracy: %f' % accuracy_af)
# precision tp / (tp + fp)
precision_af = precision_score(Y2_test, y2_pred_class_AF)
print('Precision: %f' % precision_af)
# recall: tp / (tp + fn)
recall_af = recall_score(Y2_test, y2_pred_class_AF)
print('Recall: %f' % recall_af)
# f1: 2 tp / (2 tp + fp + fn)
f1_af = f1_score(Y2_test, y2_pred_class_AF)
print('F1 score: %f' % f1_af)
# kappa
kappa_af = cohen_kappa_score(Y2_test, y2_pred_class_BF)
print('Cohens kappa: %f' % kappa_af)
# ROC AUC
auc_af = roc_auc_score(Y2_test, y2_pred_BF)
print('ROC AUC: %f' % auc_af)
'''
# confusion matrix
matrix_af = confusion_matrix(Y2_test, y2_pred_class_BF, normalize='pred')
print(matrix_af)


# Saving Stage
print("Saving history of model without filtering...")
pickle_out = open("HistoryBF_RNN.pickle", "wb")
pickle.dump(historyBF_1DCNN.history, pickle_out)
pickle_out.close()

print("Saving history of model with filtering...")
pickle_out = open("HistoryAF_RNN.pickle", "wb")
pickle.dump(historyAF_1DCNN.history, pickle_out)
pickle_out.close()
