from __future__ import print_function
import numpy as np
import csv, os
from keras import optimizers
from keras import backend as K
from keras import callbacks as C
from keras.utils import np_utils
from keras.layers.core import Lambda
from keras.layers import Dense, Flatten, MaxPool1D, MaxPool2D, Dropout, TimeDistributed, MaxPooling1D
from keras.layers import Conv2D, Input, GRU, Concatenate, BatchNormalization, Conv1D
from keras.models import Model
from matplotlib import pyplot
import matplotlib.pylab as plt

import h5py #telepíteni kell akkor is ha azt írja, hogy nem használja

# fix random seed for reproducibility
np.random.seed(123123)

#felső réteg keret es ugrás
CONV_NUM2 = 16 #filterszám

CONV_MERGE_LEN = 8
STRIDE_MERGE_LEN = 1
MAX1_SIZE = 4

CONV_MERGE_LEN3 = 8
STRIDE_MERGE_LEN3 = 1
MAX2_SIZE = 4

drivers = []
for root, dirs, files in os.walk("./probe"):
    for filename in files:
        drivers.append(os.path.join(root, filename))

print(len(drivers))


INTER_DIM = 8 #GRU dimenzió
TAU = 40 #időszelet, adatok száma
PERSONS = drivers #fájlnevek, csv-k
SENSORS = (707,)#(384, 1040, 416, 272) #szenzornevek
EPOCH_CNT = 30 #epoch szám
TRAINING = True #tanítás/beolvasás
PERCENT= 0.8 #train size/test size
BATCH_SIZE= 32
WINDOW_SIZE=2400 #super ablak méret
STEP= 10 #eltolás két super ablak között
WEIGHTS_FILE_OUT = 'weightsbest.h5py' #tanítás kimenet
WEIGHTS_FILE_IN2 = 'weightsbest.h5py' #beolvasás

def create_dataset(dataset, look_back=1, category=1):
    dataX, dataY = [], []
    for j in range(0, len(dataset) - WINDOW_SIZE + 1, STEP):
        tmp=[]
        for i in range(j, j + WINDOW_SIZE, look_back):
            a = dataset[i:(i + look_back)]
            tmp.append(a)
        dataX.append(tmp)
        dataY.append(category)

    return np.array(dataX), np.array(dataY)




def prepare(file=None, slice=TAU, category=1, classes=len(PERSONS)):
    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        keysset = set()
        datas = []
        for i in range(len(SENSORS)):
            datas.append([])
        # data = []
        # data2 = []
        for rows in reader:
            ertek = float(rows[2])
            kulcs = int(rows[1])
            keysset.add(kulcs)
            for i in range(len(SENSORS)):
                if (kulcs == SENSORS[i]):
                    datas[i].append(ertek)


    #train és test szétválasztás
    train_size = int(len(datas[0]) * PERCENT)
    trains = []
    tests = []

    for i in range(len(SENSORS)):
        train, test = datas[i][0:train_size], datas[i][train_size:len(datas[i])]
        trains.append(train)
        tests.append(test)

    #időszeletek és szenzormérések kialakítása
    trainXs = []
    testXs = []
    trainY = []
    testY = []

    for i in range(len(SENSORS)):
        trainX, trainY = create_dataset(trains[i], slice, category)
        trainX, trainY = create_dataset(trains[i], slice, category)
        testX, testY = create_dataset(tests[i], slice, category)
        trainXs.append(trainX)
        testXs.append(testX)

    #2dim -> 3dim conv1d miatt
    for i in range(len(SENSORS)):
        trainXs[i] = np.expand_dims(trainXs[i], -1)
        #trainXs[i] = np.expand_dims(trainXs[i], -1)
        testXs[i] = np.expand_dims(testXs[i], -1)
        #testXs[i] = np.expand_dims(testXs[i], -1)

    # Y kategóriavektorok létrehozása
    if (len(PERSONS)>2):
        trainY = np_utils.to_categorical(trainY, classes)
        testY = np_utils.to_categorical(testY, classes)


    return trainXs, trainY, testXs, testY

#lambda réteg tensor dimenzióbővítés
def expand_dims(x):
    return K.expand_dims(x, -1)
#átlagolás
def avg_layer(x):
    return K.mean(x, 1)


#lambda réteg kimenő tensor alakja
def expand_dims_output_shape(input_shape):
    if (len(input_shape) == 2):
        return (input_shape[0], input_shape[1], 1)
    if (len(input_shape) == 3):
        return (input_shape[0], input_shape[1],input_shape[2], 1)
    else:
        return input_shape

#személyenkénti adatok létrehozása

con_trainXs = []
con_trainY = []
con_testXs = []
con_testY= []

train_list = []
test_list = []
trY_list = []
teY_list = []

for i in range(len(PERSONS)):
    trainXs, trainY, testXs, testY = prepare(file=PERSONS[i], category=i)
    train_list.append(trainXs)
    test_list.append(testXs)
    trY_list.append(trainY)
    teY_list.append(testY)

#közös adathalmazba rendezés
con_trains = np.concatenate(train_list, axis=1)
con_trainY = np.concatenate(trY_list, axis=0)
con_tests = np.concatenate(test_list, axis=1)
con_testY= np.concatenate(teY_list, axis=0)

print('train shape', con_trains.shape, 'test shape', con_tests[0].shape, 'trY', con_trainY.shape, 'teY', con_testY.shape)
newtrain=np.concatenate(con_trains, axis=-1)
newtest=np.concatenate(con_tests, axis=-1)
print(newtrain.shape)


input= Input(shape=(int(WINDOW_SIZE/TAU), TAU, len(SENSORS)))
win_input= Input(shape=(TAU, len(SENSORS)))
sen_conv1= Conv1D(CONV_NUM2,
                  kernel_size=(CONV_MERGE_LEN),
                  strides=(STRIDE_MERGE_LEN),
                  padding='causal',
                  activation='relu')(win_input)
sen_batch1= BatchNormalization()(sen_conv1)
sen_max1 = MaxPooling1D(pool_size=MAX1_SIZE)(sen_batch1)
sen_conv3= Conv1D(CONV_NUM2,
                  kernel_size=(CONV_MERGE_LEN3),
                  strides=(STRIDE_MERGE_LEN3),
                  padding='causal',
                  activation='relu')(sen_batch1)
sen_batch3= BatchNormalization()(sen_conv3)
sen_max3 = MaxPooling1D(pool_size=MAX2_SIZE)(sen_batch3)
sen_flatten= Flatten()(sen_max3)
merged_model=Model(inputs=win_input, outputs=sen_flatten)
timed=TimeDistributed(merged_model)(input)
print(merged_model.summary())

#lamb= Lambda(expand_dims, expand_dims_output_shape)(sen_flatten)

grucell= GRU(INTER_DIM, return_sequences=True)(timed)
grucell2= GRU(INTER_DIM, return_sequences=True)(grucell)
avg= Lambda(avg_layer)(grucell2)
if (len(PERSONS)>2):
    dens= Dense(len(PERSONS), activation="softmax")(avg) #sigmoid binárisnál, softmax categorinál
else:
    dens= Dense(1, activation="sigmoid")(avg)
model=Model(inputs=input, outputs=dens)
print(model.summary())


if (len(PERSONS)>2):
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])  # rms prop # binary hogyha bináris

else:
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])  # rms prop # binary hogyha bináris

if (TRAINING):
    early_stop = C.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
    checkpoints = C.ModelCheckpoint(WEIGHTS_FILE_IN2, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    callbacks_list = [checkpoints]
    history = model.fit(newtrain, con_trainY, batch_size=BATCH_SIZE, epochs=EPOCH_CNT, validation_data=(newtest, con_testY), callbacks=callbacks_list)
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.savefig('diag_loss.png')
    pyplot.show()
    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.title('model train vs validation acc')
    pyplot.ylabel('acc')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.savefig('diag_acc.png')
    pyplot.show()

else:
    model.load_weights(WEIGHTS_FILE_OUT)
    scores= model.evaluate(newtest, con_testY, verbose=1)
    print(scores[1]*100)


