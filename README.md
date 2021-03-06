# Treino de CNN para classificação de topos de rolhas

## Introdução

O objetivo era criar uma CNN que fosse capaz de fazer a classificação de rolhas em três categorias: boas, médias e más, dependendo do estado do topo. Inicialmente procedeu-se à classificação de um dataset composto por aproximadamente 1000 imagens e depois ao treino e teste destas CNNs

## Descrição das experiências

Inicialmente procedeu-se à classificação de aproximadamente 100 imagens e ao treino de uma CNN para classificar as restante de modo mais expedito. O primeiro modelo obtido tinha uma precisão de apenas 17,19%, o que não era adequado para o fim pretendido. Inicialmente tentou-se adicionar uma dropout layer o que fez a precisão subir para 21,88%, no entanto a CNN ainda estava longe de ser considerada precisa. Finalmente foi feito um pequeno estudo dos vários otimizadores possíveis, concluindo-se que o otimizador utilizado  o otimizador "Adam") não era apropriado, logo optou-se pelo otimizador SGD com learning rate de 0,001 que disparou a precisão para 76,34%. Com esta CNN tentou classificar-se as 700 imagens restantes, no entanto o modelo só encontrou 13 imagens para a categoria boa e 2 para a categoria má, logo desistiu-se deste método e passou-se à classificação manual das imagens todas. Depois da classificação obtiveram-se 143 topos na categoria boa, 459 na categoria média e 278 na má. Foi então usado o modelo aplicado às 100 imagens iniciais e isto resultou numa precisão de 74,42% e loss de 0,5494, o que são resultados bastante satisfatórios. A título de curiosidade, dado que o número de imagens em cada categoria era bastante diferente, fez-se a experiência de escolher aleatoriamente 143 imagens de cada categoria de modo a ter o mesmo número de imagens por classe e treinar uma nova CNN com este novo dataset, de modo tentar perceber se o equilíbrio de tamanho de classes têm influência nos resultados. Esta nova CNN teve uma precisão de 70,93% e uma loss de 0,6013. Sendo que estes resultados foram ligeiramente piores que os anteriores, conclui-se que o equilíbrio do tamanho das classes não é relevante para a precisão de
uma CNN e que provavelmente o decréscimo de precisão entre modelos deveu-se até à redução do número de imagens para treinar o modelo (e como já é sabido, se não atingirmos um estado de overfitting, quantas mais imagens forem usadas para treino, melhor será o modelo).
Procedendo para a realização de modelos para datasets com fundos preto e branco, dado que o nome destas novas imagens não correspondiam ao das imagens do dataset original, teve de ser realizada de novo uma classificação manual do dataset. Como, pelo menos, as imagens com fundo preto tinham o mesmo nome que as de fundo branco para a mesma rolha, apenas foi necessário classificar o dataset de fundo preto e depois fazer a correspondência para o dataset de fundo branco. Com estes novos datasets foram treinadas duas CNNs. A CNN aplicada ao dataset de fundo branco obteve uma precisão de 83,03% e um loss de 0,3904 e a CNN aplicada ao dataset de fundo preto obteve uma precisão de 86,14% e um loss de 0,3448. Depois procedeu-se à augmentation de ambos os datasets, com as transformações adição por canal, Alpha Blending, variação de contraste, desfoque gaussiano e ruído gaussiano. Para pôr em perspetiva, depois das augmentations o dataset já compreendia 5022 imagens, 2400 más, 2208 médias e 414 boas. Prevê-se que o treino destas duas novas CNNs para os novos datasets aumentados tenha demorado aproximadamente 8 horas, logo a sua avaliação só foi feita na semana seguinte. Criou-se um novo dataset para teste destas
CNNs através da realização de mais augmentations às imagens originais. Os resultados para a CNN de fundo branco foi uma precisão de 92,35% e uma loss de 0,1858 e para a de fundo preto foi uma precisão de 96,36% e uma loss de 0,1216.

A tabela seguinte resume os resultados de todas as CNNs:

|Dataset|Precisão|Loss|
|---|---|---|
|Fundo Normal|74,42%|0,5494|
|Fundo Branco|83,03%|0,3904|
|Fundo Preto|86,14%|0,3448|
|Fundo Branco com augmentations|92,35%|0,1858|
|Fundo Preto com augmentations|96,36%|0,1216|

Conclui-se então que remover o fundo, trocá-lo por uma cor sólida e recortar a imagem para um Aspect
Ratio quadrado é efetivamente vantajoso, pois obtemos CNNs com maior precisão, provavelmente porque
o modelo não é "distraído"pelo fundo da imagem e a distorção aplicada pelo modelo às imagens no seu
processamento é menos nocivo quando a imagem é quadrada, pois preserva as proporções. É também de
notar que neste caso o uso de fundo preto foi vantajoso em relação ao branco, mas provavelmente isto
é um caso específico desta aplicação. Conclui-se também que aplicar augmentations é também muito
vantajoso pois, mesmo sendo testadas com imagens com augmentations (que funcionam como distorções)
ao contrário do modelo com fundo normal, os modelos treinados com augmentations conseguiram ainda
assim superar a performance dos modelos anteriores. Continuou também a verificar-se uma superioridade
do modelo treinado com imagens com fundo preto.

## Repetir resultados

O script usado para treinar os modelos esta dentro da pasta "classificacao", sendo possivel usa-lo especificando uma pasta onde estão as pastas correspondentes a categorias com as imagens do dataset colocadas nas respetivas pastas e o modelo, no fim de treinado, será colocado na pasta "saved_model"

# Treino de R-CNN para deteção de Bofes - 1ª tentativa

## Introdução
Inicialmente procedeu-se à tentativa de criar uma R-CNN de raiz
A RCNN criada é composta por duas componentes, uma propõe regiões de interesse e a outra classifica essas regiões de interesse.
A primeira parte usa o algoritmo _Selective Search_ da biblioteca OpenCV para criar aproximadamente 400 propostas para uma imagem que lhe é dada.
A segunda parte usa estas propostas e passa-as através de uma CNN treinada para obter a classificação destas regiões.

## Treino

A estratégia usada foi primeiro usar o algoritmo de _Selective Search_ para obter sugestões de localizações de Bofes e exportar as sugestões para duas pastas, uma com sugestões que correspondem a sugestões corretas de bofes e outra que corresponde a sugestões erradas.

Em relação à exportação foram primeiro importadas as anotações com o seguinte código:

```python
def importAnnotations(annotationPath, imageName):
    
    #Import annotation file
    file =  open(annotationPath + imageName + ".xml")
    if file == None:
        print("ERROR FAILED TO OPEN ANNOTATION FILE")
        return None

    #Converting XML file to list of bounding box coordinates    
    file_data = file.read()
    dict_data = xmltodict.parse(file_data)
    boundingBoxesRaw = []
    if "object" in dict_data["annotation"].keys():
        if str(type(dict_data["annotation"]["object"])) == "<class 'collections.OrderedDict'>":
            boundingBoxesRaw.append(dict_data["annotation"]["object"]["bndbox"])
        elif str(type(dict_data["annotation"]["object"])) == "<class 'list'>":
            for i in range(len(dict_data["annotation"]["object"])):
                boundingBoxesRaw.append(dict_data["annotation"]["object"][i]["bndbox"])
    
    #Rearranging bounding box coordinates in the correct shape
    BoundingBoxes = []
    for i in range(len(boundingBoxesRaw)):
        x1 = float(boundingBoxesRaw[i]["xmin"])
        y1 = float(boundingBoxesRaw[i]["ymin"])
        x2 = float(boundingBoxesRaw[i]["xmax"])
        y2 = float(boundingBoxesRaw[i]["ymax"])
        BoundingBoxes.append([x1, y1, x2, y2])
    
    #Return list of bounding boxes
    if len(BoundingBoxes) == 0:
        return None
    else:
        return BoundingBoxes
```

Após ter as lista de _bounding boxes_ relativa às anotações, faz-se então uma iteração por todas as imagens (no total 7700, já com augmentations), usando o seguinte código:

```python
for filename in os.listdir("images/"):
    img = cv2.imread("images/"+ filename)
```

Aplica-se-lhes então o seguinte algoritmo:

1. Usando a Selective Search obtêm-se aproximadamente 400 propostas por imagem
```python
    proposals = selectiveSearch(img)
```
2. Usando Non-maximum Supression (NMS), reduz-se estas propostas para apoximadamente 100 com o código:
```python
    proposals = NMS(proposals,NMSThreashold)
```
```python
def NMS(boxes, overlapThresh):
    result = []
    for i in range(len(boxes)):
        indexes_to_delete = []
        if i>=len(boxes):
            break
        result.append(boxes[i])
        for j in range(i+1,len(boxes)):
            boxesXY1 = xywh2xyxy(boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3])
            boxesXY2 = xywh2xyxy(boxes[j][0],boxes[j][1],boxes[j][2],boxes[j][3])
            if IoU(boxesXY1,boxesXY2) > overlapThresh:
                indexes_to_delete.append(j)
        if indexes_to_delete == []:
            continue
        boxes = arrayRemove(boxes,indexes_to_delete)
    return result
```

3. Calcula-se a _Intersection over Union_ (IoU) de todas as _bounding boxes_ com todas as sugestões e as que tem um IoU acima de um threshold são guardados na pasta "bofes" e os outros na pasta "clean"

```python
count = 0
#Loop through all proposals in an image
for proposal in xyxyProposals:
    #Get proposal coordinates in the correct form
    proposalDim = xyxy2xywh(proposal[0],proposal[1],proposal[2],proposal[3])
    #Solution for a random bug (?)
    if proposalDim[2] == 0 or proposalDim[3] == 0:
        continue
    #Extracting the image realting to a proposal
    proposalImage = img[proposalDim[1]:proposalDim[1] + proposalDim[3], proposalDim[0]:proposalDim[0] + proposalDim[2]]
    proposalImage = cv2.resize(proposalImage, (224, 224))
    #If there are no annotations for this image then the proposal is automatically considered not to be a "Bofe"
    if annotations == None:
        if not(cv2.imwrite("clean/" + filename[:-4] + str(count) + ".jpg",proposalImage)):
            print("ERROR SAVING IMAGE")
        count += 1
        continue
    for annotation in annotations:
        #If IoU is over "threshold" then save in folder "bofe" 
        if(IoU(proposal,annotation) > threshold):
            cv2.imwrite("bofe/" + filename[:-4] + str(count) + ".jpg",proposalImage)
            bofeCount+=1
            count += 1
            break
        #Else save in folder "clean"
        else:
            cv2.imwrite("clean/" + filename[:-4] + str(count) + ".jpg",proposalImage)
            count += 1
            break
imageCount += 1
```


O código completo relativo a esta primeira parte do algoritmo de treino está contido no ficheiro detecao/primeira_tentativa/selective_search_train.py, incluindo todas as funções auxiliares usadas (cujo funcionamento não vale realmente a pena explicar).

É de notar que a execução deste código é extremamente demorada, estimando-se que tenha demorado aproximadamente 8 horas a correr da primeira vez que foi testado, apesar do PC não ter uma plca gráfica.

Este código gerou 654 imagens de Bofes e 1 139 634 imagens sem Bofes (equivalente a 8Gb). Perante esta enormíssima discrepância entre número de imagens de cada classe, foram apenas selecionadas aleatoriamente 1308 imagens sem Bofes de entre as obtidas para treinar a CNN, usando o código seguinte (detecao/primeira_tentativa/clean_selection.py):

```python
import os
import random
import shutil

images = os.listdir("clean/")
random_images = random.choices(images,k = 1308)
for imageName in random_images:
    shutil.copyfile("clean/" + imageName,"clean_selected/" + imageName)
```

Com as 654 imagens de Bofes e as 1308 imagens sem Bofe foi criado o seguinte algoritmo para treino de uma CNN (detecao/primeira_tentativa/learning.py):

```python
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.applications import VGG16

num_classes = 8
input_shape = (128,128,3)

print("Importing data...","")
trainingData = tf.keras.preprocessing.image_dataset_from_directory(directory = "C:\estagio\\neural_networks\Projeto\modelo\\train",image_size=(224,224), label_mode = 'binary')
if trainingData == None:
    print("ERROR dataset not imported!")
print("Imported!")
base_model = VGG16(weights='imagenet', include_top=False)

for layer in base_model.layers[:14]:
    layer.trainable = False
for layer in base_model.layers[14:]:
    layer.trainable = True

model = keras.models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Flatten(),
    layers.Dense(1024,activation='relu'),
    layers.Dense(2,activation = 'softmax'),
    layers.Flatten()
])
model.summary()

batch_size = 128
epochs = 15

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print("Model compiled")
model.fit(x=trainingData, batch_size=batch_size, epochs=epochs)

model.save('.\modelo')
```

O modelo usado foi o seguinte:

![picture alt](https://i.imgur.com/JZOoVXm.png "Resultados do treino")

Durante o treino, os resultados foram bastante insatisfatórios pois o modelo bloqueia numa _loss_ de 7.6832 e uma _accuracy_ de 0.6667, provavelmente devido a _overfitting_. Para a próxima semana realizar-se-á a tentativa de diminuir o número de _epochs_ e de adicionar _Dropout Layers_ para tentar resolver este problema.

![picture alt](https://i.imgur.com/thqRKfg.png "Resultados do treino")


# Treino de R-CNN para deteção de Bofes - Usando YOLO

## Introdução
Dado que criar uma R-CNN de raíz teve péssimos resultados, procedeu-se à tentativa de treino de uma R-CNN recorrendo ao _transfer model_ YOLOv3.

## Procedimento
Inicialmente converteu-se as labels que estavam em formato Pascal VOC em formato YOLO recorrendo a um funções criadas anteriormente. O códio usado foi o seguinte (detecao/darknet/pascalVOCtoYOLO.py):

```python
print("Converting Pascal VOC annotations to Yolo v1.1")
imageCount = 0
annots = os.listdir("annotations/")
for filename in annots:
    boxes = importAnnotations("annotations/",filename[:-4])
    f = open("TEMP_new_annot/" + filename[:-3] + "txt","w")
    if boxes != None:
        for box in boxes:
            x = ((box[0] + box[2]) / 2) / 512
            y = ((box[1] + box[3]) / 2) / 512
            w = (box[2] - box[0]) / 512
            h = (box[3] - box[1]) / 512
            f.write("0 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")
    f.close()
    imageCount+=1
    print("{:.2f}".format(imageCount/len(annots)*100) + "% (" + str(imageCount) + "/" + str(len(annots)) + ")", end = '\r')
print("DONE!")
```

Depois, usando o repositório https://github.com/ultralytics/yolov3 foi possível, com apenas o comando

```bash
python train.py --img 512 --batch 16 --epochs 5 --data topos512.yaml --weights yolov3.pt
```

foi possível treinar um modelo de deteção recorrendo à estrutura YOLO.

(RESULTADOS AINDA EM PROCESSO)

## Reproduzir resultados

Para reproduzir os resultados, depois de ser fazer download do repositório disponível em https://github.com/ultralytics/yolov3, dentro da pasta _data_ deste é preciso adicionar o ficheiro /detecao/darknet/topos512.yaml. No mesmo diretório onde se encontra o repositório instalado (isto é o repositório acima de, por exemplo, a pasta _data_) tem de ser criada a pasta _dataset_ e tem de ser colocado lá dentro todos os ficheiros contidos na pasta /detecao/darknet/dataset deste repositório. Dado que este repositório é público e as imagens usadas para treinar os modelos descritos anteriormente são propriedade da empresa Vimétrica, estas não estão incluidas nos ficheiros deste repositório, mas teriam de ser adicionadas a dataset/topos512/obj_train_data, juntamente com as suas anotações em formato YOLOv1.1. É necessário adicionar também o nome de todas as imagens ao ficheiro dataset/topos512/train.txt.