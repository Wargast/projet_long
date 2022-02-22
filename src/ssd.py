import numpy as np
import cv2
import math
import time

im_2 = cv2.imread('../datas/middlebury/artroom1/im0.png', cv2.IMREAD_GRAYSCALE)
im_ref = cv2.imread('../datas/middlebury/artroom1/im0.png', cv2.IMREAD_GRAYSCALE)



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
    f = math.floor(taille_f/2)



    ## k = 0
    blocks_ref = np.lib.stride_tricks.sliding_window_view(im_ref,(taille_f,taille_f))
    blocks = np.lib.stride_tricks.sliding_window_view(im_2,(taille_f,taille_f))

    blocks_0 = np.ones(blocks.shape)*np.inf    
    error = np.square(blocks_ref-blocks).sum(axis=2).sum(axis=2)
    disparity = np.zeros((blocks.shape[0],blocks.shape[1]))
    for k in range(1,128):
        blocks_k = blocks_0
        # decalage de k
        print(k)
        blocks_k[:,:blocks_k.shape[1]-k,:,:] = blocks[:,k:,:,:]

        # Calcul de l'erreur  ssd
        error_k = np.square(blocks_ref-blocks_k).sum(axis=2).sum(axis=2)

        # Calcul du decalage minimum 
        disparity[error > error_k] = k
        error = np.minimum(error,error_k)

    disparity_res = np.ones((l,c))*np.inf
    disparity_res[f:l-f,f:c-f] = disparity
    
    # Création de la carte de disparités
    # for i in range(l):
    #     for j in range(c):
    #         if i > f and i < (l-f) and j > f and j < (c-f):
    #             bloc0 = im_ref[i-f:i+f,j-f:j+f]
    #             erreurs = np.zeros((1,128))


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
                
    np.save("disparity_ssd.npy", disparity_res)
    return disparity_res


debut = time.time()
ssd_disparites(im_ref=im_ref, im_2=im_2)
fin = time.time() - debut
print(fin)