import numpy as np
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import pandas


def load_only_data():
    return pandas.read_csv("train.csv")


d = load_only_data().as_matrix()

# l = 63
d = d[1:20001,:]

#
size = 0
un_v = {}
for i in range(0, 63):
    if isinstance(d[0][i], str):
        q = pandas.get_dummies(d[:, i])
        size += q.shape[1]
        un_v[i] = q.as_matrix()
    else:
        un_v[i] = d[:, i].reshape(20000,-1)
        size += 1

res = np.hstack((un_v[0], un_v[1]))

for i in range(2, len(un_v)):
    print(i)
    res = np.hstack((res, un_v[i]))

X = res[:, :-2]
y = res[:, -1]
X = np.nan_to_num(X)
X = X.astype('float64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

y_test.tolist()
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=25)

clf.fit(X_train, y_train.tolist())



batch_size = 32
nb_classes = 6
nb_epoch = 140

data_augmentation = True

print("Loading data...")
# (X_train, y_train), (X_test, y_test) = load_data(data_augmentation)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print(y_train)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train)
model= Sequential([
    Dense(20, input_shape=(X_train.shape[1], )),
    Activation('relu'),
    Dense(y_train.shape[1]),
    Activation('softmax')
])

from keras import optimizers
adam = optimizers.Adam(lr = 0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=100, batch_size=32, validation_data=(X_test, y_test))

# model = Sequential()
#
# model.add(Convolution1D(nb_filters, nb_conv,
#                         border_mode='valid',
#                         input_shape=(img_rows, img_cols)))
# model.add(Activation('relu'))
# model.add(Convolution1D(nb_filters, nb_conv))
# model.add(Activation('relu'))
# model.add(MaxPooling1D())
# # model.add(Dropout(0.15))
#
# model.add(Convolution1D(2 * nb_filters, nb_conv, border_mode='same'))
# model.add(Activation('relu'))
# model.add(Convolution1D(3 * nb_filters, nb_conv))
# model.add(Activation('relu'))
# model.add(MaxPooling1D())
# # model.add(Dropout(0.25))
#
# # New layer
# model.add(Convolution1D(3 * nb_filters, nb_conv))
# model.add(Activation('relu'))
# model.add(Convolution1D(3 * nb_filters, nb_conv))
# model.add(Activation('softplus'))
# model.add(MaxPooling1D())
# # model.add(Dropout(0.4))
#
# # model.add(Flatten())
# # model.add(Dense(244))
# # model.add(Activation('relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(nb_classes))
# # model.add(Activation('softmax'))
#
# # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# print("Done compiling")
#
# # history = LossHistory()
# history = None
# checkpointer = ModelCheckpoint(filepath="weights/weights.hdf5", verbose=1, save_best_only=True)
# earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
#
# # X_train /= 255
# # X_test /= 255
#
# history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#                      verbose=1, validation_data=(X_test, y_test))
# score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
