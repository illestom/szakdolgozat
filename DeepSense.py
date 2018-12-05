from __future__ import print_function
import numpy as np
import csv
from keras import optimizers
from keras import backend as K
from keras import callbacks as C
from keras.utils import np_utils
from keras.layers.core import Lambda
from keras.layers import Dense, Flatten, MaxPool1D, MaxPool2D, Dropout, TimeDistributed
from keras.layers import Conv2D, Input, GRU, Concatenate, BatchNormalization, Conv1D
from keras.models import Model
from matplotlib import pyplot
import matplotlib.pylab as plt
import pydot
import graphviz
import h5py

# fix random seed for reproducibility
np.random.seed(123123)



INTER_DIM = 8 #GRU dimenzió
TAU = 40 #időszelet, adatok száma
PERSONS = ('A', 'B') #fájlnevek, csv-k, kiterjesztés nélkül
SENSORS = (707, 1001)#(384, 1040, 416, 272) #szenzornevek
EPOCH_CNT = 30 #epoch szám
PERCENT= 0.8 #train size/test size
BATCH_SIZE= 128
WINDOW_SIZE=2400 #super ablak méret
STEP= 10 #eltolás két super ablak között
WEIGHTS_FILE_OUT = 'weightsbest.h5py' #tanítás kimenet
WEIGHTS_FILE_IN2 = 'weightsbest.h5py' #beolvasás



#alsó réteg keret es ugrás
CONV_NUM = 16 #filterszám

CONV_LEN_FIRST = 8
CONV_MAX_FIRST = 2
STRIDE_FIRST = 1

CONV_LEN_INTE = 4
CONV_MAX_INTE = 2
STRIDE_INTE = 1

CONV_LEN_LAST = 2
CONV_MAX_LAST= 2
STRIDE_LAST = 1

#felső réteg keret es ugrás
CONV_NUM2 = 16 #filterszám

CONV_MERGE_LEN = (8, len(SENSORS))
STRIDE_MERGE_LEN = (1, 1)
MAX1_SIZE=(5, 1)

CONV_MERGE_LEN2 = (4, 1)
STRIDE_MERGE_LEN2 = (1, 1)
MAX2_SIZE= (5, 1)

CONV_MERGE_LEN3 = (8, 1)
STRIDE_MERGE_LEN3 = (1, 1)
MAX3_SIZE= (5, 1)

#ablakok futtatása az adathalmazon
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

#adathalmaz létrehozása a fájlokból
def prepare(file=None, slice=TAU, category=1, classes=len(PERSONS)):
    with open(file+'.csv') as csvfile:
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


    #train és test arányszám
    train_size = int(len(datas[0]) * PERCENT)
    trains = []
    tests = []

    #szenzoronkénti train/test szétválasztás
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
        testX, testY = create_dataset(tests[i], slice, category)
        trainXs.append(trainX)
        testXs.append(testX)

    #2dim -> 3dim conv1d miatt
    for i in range(len(SENSORS)):
        trainXs[i] = np.expand_dims(trainXs[i], -1)
        testXs[i] = np.expand_dims(testXs[i], -1)

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

#precision definició
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#recall definició
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

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

newtrain=np.concatenate(con_trains, axis=-1)
newtest=np.concatenate(con_tests, axis=-1)


#alsó konv rétegek
def ConvLayers(split):
    exp = Lambda(expand_dims, expand_dims_output_shape)(split)
    sen1_conv1 = Conv1D(CONV_NUM,
                        kernel_size=(CONV_LEN_FIRST),
                        strides=(STRIDE_FIRST),
                        padding='valid',
                        activation='relu')(exp)
    sen1_batch1 = BatchNormalization()(sen1_conv1)
    sen1_max1 = MaxPool1D(pool_size=CONV_MAX_FIRST, strides=None, padding='valid')(sen1_batch1)
    sen1_conv2 = Conv1D(CONV_NUM,
                        kernel_size=(CONV_LEN_INTE),
                        strides=(STRIDE_INTE),
                        padding='valid')(sen1_max1)
    sen1_batch2 = BatchNormalization()(sen1_conv2)
    sen1_max2 = MaxPool1D(pool_size=CONV_MAX_INTE, strides=None, padding='valid')(sen1_batch2)
    sen1_conv3 = Conv1D(CONV_NUM,
                        kernel_size=(CONV_LEN_LAST),
                        strides=(STRIDE_LAST),
                        padding='valid',
                        activation='relu')(sen1_max2)
    sen1_batch3 = BatchNormalization()(sen1_conv3)
    sen1_max3 = MaxPool1D(pool_size=CONV_MAX_LAST, strides=None, padding='valid')(sen1_batch3)
    sen1_flatten = Flatten()(sen1_max3)
    lamb1 = Lambda(expand_dims, expand_dims_output_shape)(sen1_flatten)

    return lamb1

#modell összeállítása
input= Input(shape=(int(WINDOW_SIZE/TAU), TAU, len(SENSORS)))
win_input= Input(shape=(TAU, len(SENSORS)))

lambs=[]
for i in range(len(SENSORS)):
    out = Lambda(lambda x: x[:, :, i])(win_input)
    lambs.append(ConvLayers(out))

merg= Concatenate(2)(lambs)

lamb= Lambda(expand_dims, expand_dims_output_shape)(merg)

sen_conv1= Conv2D(CONV_NUM2,
                  kernel_size=(CONV_MERGE_LEN),
                  strides=(STRIDE_MERGE_LEN),
                  padding='valid',
                  activation='relu')(lamb)
sen_batch1= BatchNormalization()(sen_conv1)
sen_max1 = MaxPool2D(pool_size=MAX1_SIZE)(sen_batch1)
sen_conv2= Conv2D(CONV_NUM2,
                  kernel_size=(CONV_MERGE_LEN2),
                  strides=(STRIDE_MERGE_LEN2),
                  padding='valid',
                  activation='relu')(sen_max1)
sen_batch2= BatchNormalization()(sen_conv2)
sen_max2 = MaxPool2D(pool_size=MAX2_SIZE)(sen_batch2)
sen_conv3= Conv2D(CONV_NUM2,
                  kernel_size=(CONV_MERGE_LEN3),
                  strides=(STRIDE_MERGE_LEN3),
                  padding='valid',
                  activation='relu')(sen_batch2)
sen_batch3= BatchNormalization()(sen_conv3)
sen_max3 = MaxPool2D(pool_size=MAX3_SIZE)(sen_batch3)
sen_flatten= Flatten()(sen_batch3)

merged_model=Model(inputs=win_input, outputs=sen_flatten)
timed=TimeDistributed(merged_model)(input)
#elkészült conv struktúra
print(merged_model.summary())

grucell= GRU(INTER_DIM, return_sequences=True)(timed)
grucell2 = GRU(INTER_DIM, return_sequences=True)(grucell)
avg= Lambda(avg_layer)(grucell2)
if (len(PERSONS)>2):
    dens= Dense(len(PERSONS), activation="softmax")(avg) #sigmoid binárisnál, softmax categorinál
else:
    dens= Dense(1, activation="sigmoid")(avg)
model=Model(inputs=input, outputs=dens)
#teljes modell
print(model.summary())

if (len(PERSONS)>2):
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])  # rms prop # binary hogyha bináris

else:
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy', recall, precision])  # rms prop # binary hogyha bináris


#futás közben a legjobb eredmény kiválasztása és mentése, diagrammok létrehozása
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
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.title('model train vs validation acc')
pyplot.ylabel('acc')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.savefig('diag_acc.png')

model.load_weights(WEIGHTS_FILE_OUT)
scores= model.evaluate(newtest, con_testY, verbose=1)
print(scores[1]*100)