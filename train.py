
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import keras
from keras import backend as K
import numpy as np
import cv2
ALPHA = 1.0


NCLASSES = 2
HEIGHT = 512
WIDTH = 512
NumChannels=3
batch_size = 2
load_pretrained_weights=0
n=4                   #n: rate of codebook 1/n
C=2                   #C: rate of reduction channels 1/C
log_dir = "logs3/"

def generate_arrays_from_file(lines,batch_size):
    #
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        #
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            #
            """
            img = Image.open(r"/test3/image" + '/' + name)

            img = np.array(img)
            img = img/255
            X_train.append(img)
            """
            img=rasterio.open(r"/test3/image" + '/' + name)


            img=np.moveaxis(img.read()[0:3],0,2)
            #img = img/255
            X_train.append(img)

            name = (lines[i].split(';')[1]).replace("\n", "")

            label = cv2.imread(r"/content/drive/MyDrive/test3/label" + '/' + name)/255

            label = np.array(label)



            seg_labels = np.zeros((int(HEIGHT),int(WIDTH),NCLASSES))
            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (label[:,:,0] == c ).astype(int)
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
            Y_train.append(seg_labels)

            #
            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))

def loss(y_true, y_pred):
    loss = K.binary_crossentropy(y_true,y_pred) #K.categorical_crossentropy(y_true,y_pred)
    return loss




if __name__ == "__main__":

    #
    model = model (HEIGHT, WIDTH, NumChannels, NCLASSES,n,C)



    #model.load_weights(weights_path,by_name=True)
    model.load_weights('logs3/*ep021-loss0.072-val_loss0.022.h5')


    # fine-tune model (train only last conv layers)
    if load_pretrained_weights:
      flag = 0
      print("START FINE TUNE")
      for k, l in enumerate(model.layers):
           l.trainable = False
           if l.name == 'fine_tune1'or'fine_tune2'or'fine_tune3': #'fine_tune1'or
              flag = 1
           if flag:
              l.trainable = True

    # txt
    with open(r"/content/drive/MyDrive/test3/train3.txt","r") as f:
        lines = f.readlines()

    #
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    #
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val  #len(lines)

    #
    checkpoint_period = ModelCheckpoint(
                log_dir +'*'+ 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                        monitor='loss', #'val_loss'
                        save_weights_only=True,
                        save_best_only=True,
                        period=1
                              )
    #
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.2,
                            patience=5,
                            verbose=1
                        )
    #
    early_stopping = EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=3,
                            verbose=1
                        )

    #
    model.compile(loss = loss,
            optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),   #1e-3
            metrics = ['binary_accuracy'])


    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    #
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:],batch_size), #(lines[num_train:]batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr])