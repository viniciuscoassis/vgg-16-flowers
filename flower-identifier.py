import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

num_classes = 2
image_size = 224
batch_size_training = 100
batch_size_validation = 100

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

from google.colab import drive
drive.mount('./identification', force_remount=True)

train_generator = data_generator.flow_from_directory(
  './identification/train',
  target_size=(image_size,image_size),
  batch_size=batch_size_training,
  class_mode='categorical'
)

valid_generator = data_generator.flow_from_directory(
  './identification/valid',
  target_size=(image_size,image_size),
  batch_size=batch_size_training,
  class_mode='categorical'
)

model = Sequential()

model.add(
  VGG16(
      include_top=False,
      pooling='avg',
      weights='imagenet',
      input_shape=(image_size, image_size, 3)
  )
)


model.add(Dense(num_classes, activation='softmax'))
model.layers

model.layers[0].layers

model.layers[0].trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
steps_per_epoch_train = len(train_generator)
steps_per_epoch_valid = len(valid_generator)
number_epochs=2

fit_history = model.fit_generator(
  train_generator,
  steps_per_epoch=steps_per_epoch_train,
  epochs=number_epochs,
  verbose=1,
  validation_data=valid_generator,
  validation_steps=steps_per_epoch_valid
)

model.save('./vgg16.h5')

from keras.models import load_model
vgg16_saved = load_model('./vgg16.h5')

test_gen = data_generator.flow_from_directory(
  './identification/test',
  target_size=(image_size,image_size),
  shuffle=False
)
steps_per_epoch_test = len(test_gen)

test_history =vgg16_saved.evaluate_generator(test_gen, steps_per_epoch_test, verbose=1)
print("Accuracy for testing is: ", test_history[1])

predict_vgg16= vgg16_saved.predict_generator(test_gen, steps_per_epoch_test, verbose=1)

for i in range(0, len(predict_vgg16)):
  neg = predict_vgg16[i][0]
  pos = predict_vgg16[i][1]
  if(neg > pos):
    print("Negative")
  else:
    print("Positive")


from keras.preprocessing import image
import numpy as np

def predict_image(img_path):
  img = image.load_img(img_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = preprocess_input(img_array)
  prediction = vgg16_saved.predict(img_array)
  return prediction

img_path = './identification/test/positive/18622672908_eab6dc9140_n_jpg.rf.89def480591c1c229e7e416383dadbf3.jpg'
prediction = predict_image(img_path)
print(f'Prediction: {"Positive" if prediction[0][1] > 0.5 else "Negative"}')