import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.morphology import reconstruction


"""
Filtering regional maxima
Ejemplo en scikit image web site

"""

def RegionalMaxima(imagen,sgm,h):  
    #########################################################################
    # Convert to float: Important for subtraction later which won't work
    # with uint8
    image = img_as_float(imagen)
    image = gaussian_filter(image, sgm)
    
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    
    dilated = reconstruction(seed, mask, method='dilation')
    
    ######################################################################
    # Subtracting the dilated image leaves an image with just the figure 
    #and a flat, black background, as shown below.
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1,
                                        ncols=3,
                                        figsize=(8, 2.5),
                                        sharex=True,
                                        sharey=True)
    
    ax0.imshow(image, cmap='gray')
    ax0.set_title('original image')
    ax0.axis('off')
    
    ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
    ax1.set_title('dilated')
    ax1.axis('off')
    
    ax2.imshow(image - dilated, cmap='gray')
    ax2.set_title('image - dilated')
    ax2.axis('off')
    
    fig.tight_layout()
    
    # ######################################################################
    # # Although the features are clearly isolated surrounded by a bright 
    # # background in the original image are dimmer in the
    # # subtracted image. We can attempt to correct this using a different
    # # seed image.
    # #
    # # Instead of creating a seed image with maxima along the image border, we can
    # # use the features of the image itself to seed the reconstruction process.
    # # Here, the seed image is the original image minus a fixed value, ``h``.
    
    #h = 0.4
    seed = image - h
    dilated = reconstruction(seed, mask, method='dilation')
    hdome = image - dilated
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.5))
    yslice = 197
    
    ax0.plot(mask[yslice], '0.5', label='mask')
    ax0.plot(seed[yslice], 'k', label='seed')
    ax0.plot(dilated[yslice], 'r', label='dilated')
    ax0.set_ylim(-0.2, 2)
    ax0.set_title('image slice')
    ax0.set_xticks([])
    ax0.legend()
    
    ax1.imshow(dilated, vmin=image.min(), vmax=image.max(), cmap='gray')
    ax1.axhline(yslice, color='r', alpha=0.4)
    ax1.set_title('dilated')
    ax1.axis('off')
    
    ax2.imshow(hdome, cmap='gray')
    ax2.axhline(yslice, color='r', alpha=0.4)
    ax2.set_title('image - dilated')
    ax2.axis('off')
    
    fig.tight_layout()
    plt.show()
    
    #Grafica Comparacion
    fig, axes = plt.subplots(1,2, figsize=(8, 4))
    ax = axes.ravel()
    
    ax[0].imshow(imagen)
    ax[0].set_title("Original")
    ax[1].imshow(hdome, cmap=plt.cm.gray)
    ax[1].set_title("Regional Maxima")
    
    fig.tight_layout()
    plt.show()
    return hdome