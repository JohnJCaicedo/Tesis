from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def rgb2grascale(imagen):
    
    #RGB a GrayScale
    grayscale = rgb2gray(imagen)
    
    #Grafica Comparacion
    # fig, axes = plt.subplots(1,2, figsize=(8, 4))
    # ax = axes.ravel()
    
    # ax[0].imshow(imagen)
    # ax[0].set_title("Original")
    # ax[1].imshow(grayscale, cmap=plt.cm.gray)
    # ax[1].set_title("Grayscale")
    
    # fig.tight_layout()
    # plt.show()
    return grayscale
