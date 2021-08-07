from skimage import feature
import matplotlib.pyplot as plt


#####################################################################
"""
#Canny edge detector Exmaple
#Recomendado usar constrast stretching
#grayscale 
#img_rescale 
"""

def Edges(im,sgm1,sgm2):   
    # Compute the Canny filter for two values of sigma
    edges1 = feature.canny(im,sigma=sgm1)
    edges2 = feature.canny(im, sigma=sgm2)
    
    # # display results
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
    #                                     sharex=True, sharey=True)
    # ax1.imshow(im, cmap=plt.cm.gray)
    # ax1.axis('off')
    # ax1.set_title('noisy image', fontsize=20)    
    # ax2.imshow(edges1, cmap=plt.cm.gray)
    # ax2.axis('off')
    # ax2.set_title(f'Canny filter, $\sigma=${sgm1}', fontsize=20)    
    # ax3.imshow(edges2, cmap=plt.cm.gray)
    # ax3.axis('off')
    # ax3.set_title(f'Canny filter, $\sigma=${sgm2}', fontsize=20)
    # fig.tight_layout()    
    # plt.show()
    return edges1