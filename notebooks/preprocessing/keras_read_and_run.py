
# coding: utf-8
import os
data_folder = '../data'
images_folder = os.path.join(data_folder,'new_roof_images/resized')
ids_path = '../data/new_roof_images/id_train_resampled.csv'
#gpu run 
#    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
from dataset import Dataset
dataset = Dataset(images_folder, ids_path=ids_path)
#dataset.ids = dataset.ids[:100]
dataset.read_dataset()

def func(row):
    return row.T

dataset.apply_each_X_data_row(func)


print('X_train shape:', dataset.X_train.shape)
print(dataset.X_train.shape[0], 'train samples')
print(dataset.X_test.shape[0], 'test samples')

batch_size = 128
nb_epoch = 12
from keras.models import model_from_json
jmodel = ""
with open('model.json') as f:
    jmodel = f.read()
model = model_from_json(jmodel)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(dataset.X_train, dataset.Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(dataset.X_test, dataset.Y_test))
score = model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
