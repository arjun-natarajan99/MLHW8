"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2020 Mar 04
Description : Image Posterization
"""

# python libraries
import sys
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# scikit-learn libraries
from sklearn.cluster import KMeans

######################################################################
# functions
######################################################################

def posterize(img, k, random_state=42):
    """Posterize image.
    
    Parameters
    --------------------
        img          -- numpy array of size (h,w,d), image data
        k            -- int, number of colors to posterize by
        random_state -- int, random state for KMeans
    
    Return
    --------------------
        img2         -- numpy array of size (h,w,d), posterized image data
    """
    
    # flatten image, features are RGB value of each pixel
    h,w,d = img.shape    # height, width, depth
    pixels = img.reshape((w*h, d))
    
    ### ========== TODO : START ========== ###
    # part a : posterize
    # professor's solution : 7 lines
    # 
    # this is just filler code, you will need to set pixels2 properly
    # for the autograder to work, make sure you initialize KMeans with random_state
    # hint : given a fitted KMeans object, use attributes cluster_centers_ and labels_
    
    pixels2 = pixels
    ### ========== TODO : END ========== ###
    
    np.clip(pixels2, 0, 1, out=pixels2) # clip to [0,1]
    img2 = pixels2.reshape(img.shape)
    
    return img2


######################################################################
# main
######################################################################

def main(argv):
    # get input
    fn = argv[0]
    k = int(argv[1])
    
    # make output filename
    root, ext = os.path.splitext(fn)
    fn2 = root + str(k) + ext
    
    # read image using RGB color mode
    img = plt.imread(fn)[:,:,:3]    # ignore alpha
    
    # posterize
    img2 = posterize(img, k)
    
    # save to file
    plt.imsave(fn2, img2, vmin=0, vmax=1)
    
    # plot
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    
    plt.imshow(img2)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])