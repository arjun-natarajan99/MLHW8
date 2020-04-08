"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2020 Mar 25
Description : Famous Faces
"""

# python libraries
import math

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# scikit-learn libraries
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

######################################################################
# LFW globals
######################################################################

LFW_IMAGESIZE = (50,37)


######################################################################
# plot functions
######################################################################

def show_image(im, rescale=False) :
    """
    Open a new window and display the image.
    
    Parameters
    --------------------
        im      -- numpy array of shape (d,), image
        rescale -- bool, whether to scale the image so that colormap covers complete value range
    """
    
    plt.figure()
    im = im.copy()
    im.resize(*LFW_IMAGESIZE)
    if rescale :
        plt.imshow(im.astype(float), cmap=plt.cm.get_cmap("gray"))
    else :
        plt.imshow(im.astype(float), cmap=plt.cm.get_cmap("gray"), vmin=0, vmax=255)
    plt.axis('off')
    plt.show()


def plot_gallery(images, rescale=False, title=None, subtitles=[], n_row=4, n_col=5):
    """
    Plot array of images.
    
    Parameters
    --------------------
        images       -- numpy array of shape (12,d), images (one per row)
        rescale      -- bool, whether to scale the image so that colormap covers complete value range
        title        -- str, title for entire plot
        subtitles    -- list of 12 strings or empty list, subtitles for subimages
        n_row, n_col -- ints, number of rows and columns for plot
    """
    
    plt.figure(figsize=(1.8*n_col, 2.4*n_row))
    if title:
        plt.suptitle(title, size=16)
    for i, comp in enumerate(images) :
        plt.subplot(n_row, n_col, i+1)
        if rescale :
            plt.imshow(comp.reshape(LFW_IMAGESIZE), cmap=plt.cm.get_cmap("gray"))
        else :
            plt.imshow(comp.reshape(LFW_IMAGESIZE), cmap=plt.cm.get_cmap("gray"), vmin=0, vmax=255)
        if subtitles:
            plt.title(subtitles[i], size=12)
        plt.axis('off')
    plt.show()


def plot_scores(krange, homogeneity, completeness, v_measure, silhouette,
                xlabel=None, title=None):
    """Plot scores.
    
    Parameters
    --------------------
        krange       -- list, input range over which scores are computed
        homogeneity  -- list, homogeneity scores
        completeness -- list, completeness scores
        v_measure    -- list, v-measure scores
        silhouette   -- list, silhouette scores
        xlabel       -- str, x-axis label
        title        -- str, title
    """
    
    # round up to nearest tenth
    def roundup(x):
        return int(math.ceil(x * 10)) / 10.0
    
    f, ax = plt.subplots(4, 1, sharex=True)
    if title:
        f.suptitle(title, size=16)
    
    # homogeneity
    ax[0].plot(krange, homogeneity)
    ax[0].set_ylim((0, roundup(max(homogeneity))))
    ax[0].set_ylabel("{}".format("homogeneity"))
    
    # completeness
    ax[1].plot(krange, completeness)
    ax[1].set_ylim((0, roundup(max(completeness))))
    ax[1].set_ylabel("{}".format("completeness"))
    
    # v-measure
    ax[2].plot(krange, v_measure)
    ax[2].set_ylim((0, roundup(max(v_measure))))
    ax[2].set_ylabel("{}".format("v-measure"))
    
    # silhouette
    ax[3].plot(krange, silhouette)
    ax[3].set_ylim((-roundup(-min(silhouette)),
                     roundup(max(silhouette))))  # special limits for silhouette score
    ax[3].set_ylabel("{}".format("silhouette"))
    
    # x-axis
    ax[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    if xlabel:
        ax[-1].set_xlabel(xlabel)
    
    f.subplots_adjust(hspace=0.1) # bring subplots closer to each other
    for axis in ax: # hide x labels and tick labels for all but bottom plot
        axis.label_outer()
    #plt.tight_layout()
    
    plt.show()


######################################################################
# helper functions
######################################################################

def limit_pics(X, y, classes=None, nim=None) :
    """
    Select subset of images from dataset.
    User can specify desired classes and desired number of images per class.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), features
        y       -- numpy array of shape (n,), targets
        classes -- list of ints, subset of target classes to retain
                   if None, retain all classes
        nim     -- int, number of images desired per class
                   if None, retain all images
    
    Returns
    --------------------
        X1      -- numpy array of shape (nim * len(classes), d), subset of X
        y1      -- numpy array of shape (nim * len(classes),), subset of y
    """
    
    n, d = X.shape
    if classes is None : classes = np.unique(y)
    if nim is None : nim = n
    
    num_classes = len(classes)
    X1 = np.zeros((num_classes*nim, d), dtype=float)
    y1 = np.zeros(num_classes*nim, dtype=int)
    
    index = 0
    for ni, i in enumerate(classes) :      # for each class
        count = 0                          # count how many samples in class so far
        for j in range(n):                 # look over data
            if count < nim and y[j] == i : # element of class
                X1[index] = X[j]
                y1[index] = ni
                index += 1
                count += 1
    
    X1 = X1[:index,:]
    y1 = y1[:index]
    return X1, y1


######################################################################
# main
######################################################################

def main() :
    # load LFW dataset
    X = np.loadtxt("../data/faces_features.csv", delimiter=",")
    y = np.loadtxt("../data/faces_labels.csv", dtype=int, delimiter=",")
    names = np.loadtxt("../data/faces_names.csv", dtype=str, delimiter=",")
    
    
    
    #================================================================================
    # explore dataset
    
    # get indices for the first 12 unique faces
    # then display sample image per person
    X_uniq, y_uniq = limit_pics(X, y, nim=1)
    plot_gallery(X_uniq)
    
    # display ``average'' image across all people
    show_image(np.mean(X,0))    # average each column
    
    ### ========== TODO : START ========== ###
    # show single image using show_image(...)
    # show several images using plot_gallery(...)
    # set rescale=False (default) for faces, which assumes pixel values of [0,255]
    # set rescale=True for eigenfaces, which scales pixel values to use full colormap
    
    
    # part a : display ``average'' image per person
    # professor's solution : 7 lines
    # hint : use limit_pics(...), set classes to Python array with single element
    
    
    # part b : display eigenfaces
    # professor's solution : 2 lines + uncomment the plot_gallery(...) call
    
    #plot_gallery([pca.components_[i,:] for i in range(12)], rescale=True, n_row=3, n_col=4)
    
    
    # part c : explore lower-dimensional representations
    # professor's solution : 8 lines
    
    
    # part d: compute number of components needed to capture at least 99% of variance
    # professor's solution : 8 lines
    # hint : we only need to fit PCA once
    #        look at the attributes of the fitted pca object
    #        use np.cumsum(...)
    
    
    ### ========== TODO : END ========== ###
    
    
    
    #================================================================================
    # explore effect of various parameters on clustering performance
    
    # limit to 4 individuals
    X4, y4 = limit_pics(X, y, [4,6,13,16], 40)
    X4_uniq, y4_uniq = limit_pics(X4, y4, nim=1)
    plot_gallery(X4_uniq, n_row=1, n_col=4)
    
    ### ========== TODO : START ========== ###
    # get scores from metrics library
    # plot performance using plot_scores(...)
    
    
    # part f : effect of number of clusters
    # professor's solution : 14 lines
    
    np.random.seed(1234)
    
    
    # part g : effect of lower-dimensional representations
    # professor's solution : 8 lines (find k*) + 18 lines (transform, cluster, and score)
    
    np.random.seed(1234)
    
    
    
    ### ========== TODO : END ========== ###
    
    
    
    #================================================================================
    # determine ``most discriminative'' and ``least discriminative'' pairs of images
    
    ### ========== TODO : START ========== ###
    # part h
    # professor's solution : 25 lines (approach) + 20 lines (plot)
    
    np.random.seed(1234)
    
    
    
    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()