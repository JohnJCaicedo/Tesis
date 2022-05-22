import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from AttributeDataset import *
import numpy as np
import matplotlib.pyplot as plt
import time
inicio = time.time()
 
cam = cv2.VideoCapture(0)
result, image_np = cam.read()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

attributes_file = 'metadata.csv'   
# attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
attributes = AttributesDataset(attributes_file)
    
# during validation we use only tensor and normalization transforms
val_transform = transforms.Compose([
transforms.Resize((120,120)),
transforms.ToTensor()])  

classes = ('Clase 1', 'Clase 2', 'Clase 3', 'Clase 4')
Tamano = ''

def init_net():

  #pretrained Alexnet
  #model = torchvision.models.alexnet(pretrained=True)
  #model.classifier[6] = nn.Linear(4096,4)

  #pretrained Resnet18
  model = torchvision.models.resnet18(pretrained=False) 
  model.fc = nn.Linear(512,4)

  #pretrained VGG19
  #model = torchvision.models.vgg19(pretrained=True) 
  #model.classifier[6] = nn.Linear(4096,4)

  #pretrained VGG11
  #model = torchvision.models.vgg11(pretrained=True) 
  #model.classifier[6] = nn.Linear(4096,4)

  return model

model = init_net()

#Load Model on the same CNN architecture as it was trained
Modelo = torch.load('C:/Users/Johnj/Desktop/Tesis/Resnet18_OPTIMIZADA(Train 0.83,Test 0.77)', map_location=torch.device('cpu'))
model = init_net()
model.load_state_dict(Modelo)
model.eval()

transformer = transforms.Compose([
                                transforms.Resize((120,120)),
                                transforms.ToTensor()])


# ##TAMAÑO
LimiteInferior = np.array([73,0,0],np.uint8)
LimiteSuperior = np.array([255,255,168],np.uint8)
mask = cv2.inRange(image_np,LimiteInferior,LimiteSuperior)

## Morfologia ##
kernel = np.ones((29,29), np.uint8)
dilated = cv2.dilate(mask,kernel)
median  = cv2.medianBlur(dilated, 21) # Add median filter to image

contornos, _ = cv2.findContours(median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# plt.imshow(median)
# plt.title("Filtro Binario")

# fig = plt.figure(figsize=(10, 7))
# fig.add_subplot(2, 2, 1)
# plt.imshow(image_np)
# plt.title("Pred Graph")
    
if result:      
    
    ###PREDICCION CLASE
    image = np.expand_dims(image_np, axis=0)
    image = torch.Tensor(image).permute(0, 3, 1, 2).to(device)
    input = Variable(image)
    output = model(input)
    index = output.data.numpy().argmax()
    pred = classes[index]  
    
    
    ##PREDICCION TAMAÑO
    c = len(contornos)
    perimetro = []
    for i in range(c):
      perimeter = cv2.arcLength(contornos[i],True)
      perimetro.append(perimeter)
    
    max_value = max(perimetro)
    #print(max_value)
    
    if (max_value >= 9000):
      Tamano = 'Muy Grande'
      #print('Muy Grande')
    if (max_value >= 6345 and max_value < 9000):
      Tamano = 'Grande'
      #print('Grande')
    if (max_value >= 5095 and max_value < 6345):
      Tamano = 'Mediana'
      #print('Mediana')

    # cv2.drawContours(image_np, contornos,-1, (0,0,255), 3)
    # fig = plt.figure(figsize=(10, 7))
    # fig.add_subplot(2, 2, 1)
    # plt.imshow(image_np)
    # plt.title("Perimetro")    
    
    cv2.imwrite('C:/Users/Johnj/Desktop/Tesis/Predicciones/{}_{}.jpg'.format(pred,Tamano), image_np)
    
else:    
	print("No image detected. Please! try again")
    
    
fin = time.time()
print(fin-inicio) # 1.0005340576171875