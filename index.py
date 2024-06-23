import numpy as np
import pandas as pd
import os
import glob
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import img_to_array, load_img
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

# Caminho dos pesos pré-treinados
weights = './vgg16.h5'

# Carregando as imagens de treino
folders = glob.glob('./train/*')
imagenames_list = []
for folder in folders:
    for f in glob.glob(folder + '/*.jpg'):
        imagenames_list.append(f)

# Função para obter o rótulo da imagem
def label_img(image):
    word_label = image.split('/')[2]
    if word_label == 'margarida':
        return [1, 0, 0]
    elif word_label == 'dente_de_leao':
        return [0, 1, 0]
    else:
        return [0, 0, 1]

# Preparando os dados de treino
train = []
for image in imagenames_list:
    label = label_img(image)
    train.append([np.array(cv2.resize(cv2.imread(image), (224, 224))), np.array(label)])
    np.random.shuffle(train)

X = np.array([i[0] for i in train])
X = X / 255
Y = np.array([i[1] for i in train])

# Criando o modelo
model = Sequential()
model.add(VGG16(include_top=False, weights=weights, input_shape=(224, 224, 3)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Treinando o modelo
# model.fit(X, Y, epochs=10, batch_size=32, validation_split=0.2)  # Ajuste os parâmetros conforme necessário

# Função para carregar e prever a imagem selecionada
def load_and_predict_image():
    filepath = filedialog.askopenfilename()
    if filepath:
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        
        if predicted_class == 0:
            result_text = "Margarida"
        elif predicted_class == 1:
            result_text = "Dente-de-Leão"
        else:
            result_text = "Outra"
        
        img = Image.open(filepath)
        img = img.resize((200, 200), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        
        image_label.configure(image=img)
        image_label.image = img
        result_label.config(text=f"Classificação: {result_text}")

# Configurando a interface gráfica
root = Tk()
root.title("Classificador de Imagens")

upload_button = Button(root, text="Selecionar Imagem", command=load_and_predict_image)
upload_button.pack()

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="Classificação:")
result_label.pack()

root.mainloop()
