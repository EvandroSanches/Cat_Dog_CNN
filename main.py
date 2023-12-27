import keras.optimizers.schedules
from keras.utils import image_dataset_from_directory
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPool2D
from keras.layers import RandomFlip, RandomZoom, RandomRotation
from keras.layers.experimental.preprocessing import Rescaling
from keras.models import Sequential, load_model
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np


epochs = 200
batch_size = 8

def CarregaDados():
    #Carregando imagens do dataset
    img_treino = image_dataset_from_directory('dataset/training_set', image_size=(64,64), batch_size=5000)

    #Definindo numpy array de dados e classes
    dados = np.empty([0,64,64,3])
    classes = np.empty((0,1))

    #Convertendo objeto tensorflow para numpy array
    for image, label in img_treino:
        dados = np.append(dados, values=image, axis=0)
        label = np.expand_dims(label, axis=1)
        classes = np.append(classes, label, axis=0)

    # Classe
    # 0 - Cachorro
    # 1 - Gato
    return dados, classes

def CriaRede():
    #Aplicando camada de pre-processamento na crianção da rede neural
    modelo = Sequential([
        Rescaling(scale=1./255,  input_shape=(64, 64, 3)),
        RandomFlip('horizontal'),
        RandomRotation(0.2),
        RandomZoom(0.3)
    ])

    #Primeira camada convolucional
    modelo.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D((2,2)))

    #Segunda camada convolucional
    modelo.add((Conv2D(filters=64, kernel_size=(3,3), activation='relu')))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D(2,2))

    #Terceira camada convolucional
    modelo.add((Conv2D(filters=128, kernel_size=(3,3), activation='relu')))
    modelo.add(BatchNormalization())
    modelo.add(MaxPool2D(2,2))

    modelo.add(Flatten())

    #Camada densa
    modelo.add(Dense(units=140, activation='relu'))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(units=140, activation='relu'))
    modelo.add(Dropout(0.4))
    modelo.add(Dense(units=1, activation='sigmoid'))


    modelo.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return modelo

def Treinamento():
    #Carrega dados de treino
    dados, classes = CarregaDados()

    #Cria modelo da rede
    modelo = CriaRede()

    #Treina rede
    result = modelo.fit(dados, classes, epochs=epochs, batch_size=batch_size)

    #Salva modelo
    modelo.save('Modelo.0.1')

    #Resultado de trinamento
    plt.plot(result.history['loss'])
    plt.title('Relação de Perda')
    plt.xlabel('Epocas')
    plt.ylabel('Perda')
    plt.show()

    plt.plot(result.history['accuracy'])
    plt.title('Relação de Acuracia')
    plt.xlabel('Epocas')
    plt.ylabel('Taxa de Acerto')
    plt.show()


def Treinamento_val_cross():
    #Treinamento em validação cruzada para analise de dados
    dados, classes = CarregaDados()

    modelo = KerasClassifier(build_fn=CriaRede, batch_size=batch_size, epochs=epochs)

    result = cross_val_score(estimator=modelo, X=dados, y=classes, cv=10)

    print('Resultado:'+str(result))
    print('Média:'+str(result.mean()))
    print('Desvio Padrão:'+str(result.std()))


def Previsao(caminho):
    #Predição de imagem passando caminho da imagem como parametro
    modelo = load_model('Modelo.0.1')

    imagem = image.load_img(caminho, target_size=(64,64))
    imagem = image.img_to_array(imagem)

    imagem = np.expand_dims(imagem, axis=0)

    previsao = modelo.predict(imagem)

    if previsao > 0.5:
        return 'Gato'
    else:
        return 'Cachorro'



result = Previsao('Dataset/training_set/gato/cat.43.jpg')
print(result)
