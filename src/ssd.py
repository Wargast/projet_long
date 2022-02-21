import numpy as np
import cv2
import math
import time

im_2 = cv2.imread('/home/apigneux/nosave/A3/projet_long/datas/middlebury/artroom1/im0.png', cv2.IMREAD_GRAYSCALE)
im_ref = cv2.imread('/home/apigneux/nosave/A3/projet_long/datas/middlebury/artroom1/im0.png', cv2.IMREAD_GRAYSCALE)



def ssd_disparites(im_ref, im_2,taille_f = 15):
    """AI is creating summary for ssd_disparites

    Args:
        im_ref ([type]): [description]
        im_2 ([type]): [description]
        taille_f (int, optional): taille de la fenetre. Defaults to 15.
    """

    # Paramètres

    l = im_ref.shape[0]
    c = im_ref.shape[1]
    disparity = np.zeros((l,c))
    f = math.floor(taille_f/2)
    blocks = np.lib.stride_tricks.sliding_window_view(im_2,(taille_f,taille_f))
    print(blocks.shape)
    
    # Création de la carte de disparités
    for i in range(l):
        for j in range(c):
            if i > f and i < (l-f) and j > f and j < (c-f):
                bloc0 = im_ref[i-f:i+f,j-f:j+f]
                erreurs = np.zeros((1,128))


    # for i in range(l):
    #     for j in range(c):
            
    #         if i > f and i < (l-f) and j > f and j < (c-f):
    #             # Par bloc avec taille_fenetre = 15
    #             bloc0 = im_ref[i-f:i+f,j-f:j+f]
    #             erreurs = np.zeros((1,128))
                
    #             for k in range(j,j+128):
    #                 if k>7 and k<(c-7):
    #                     val = np.sum(np.square(im_2[i-f:i+f,k-f:k+f]-bloc0))
    #                     np.append(erreurs, val)
                        
    #                 else:
    #                     np.append(erreurs,np.inf)       
    #             index_erreur = np.argmin(erreurs)
    #             disparity[i,j] = index_erreur - j
            
    #         else: 
    #             disparity[i,j] = np.inf
                
    np.save("disparity_ssd.npy", disparity)

debut = time.time()
ssd_disparites(im_ref=im_ref, im_2=im_2)
fin = time.time() - debut
print(fin)