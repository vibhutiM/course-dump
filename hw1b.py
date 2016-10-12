import os
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from PIL import Image

import theano
import theano.tensor as T
import re
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

theano.config.floatX='float32'
'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def reconstructed_image(D,c,num_coeffs,X_mean,im_num):
    
    '''
    This function reconstructs an image X_recon_img given the number of
    coefficients for each image specified by num_coeffs
    '''
    
    '''
        Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mean: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Integer
        an integer that specifies the number of top components to be
        considered while reconstructing
        

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    c_im = c[:num_coeffs,im_num]
    D_im = D[:,:num_coeffs]
    #c_im = c[:num_coeffs,n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
    #D_im = D[:,:num_coeffs]
    X1=np.dot(D_im,c_im)
    X2=(X1.T).reshape(X_mean.shape[0],X_mean.shape[1])+X_mean
    X3=X2.flatten()
    X_recon_img=X3.reshape(256,256)
    
    
    
    

    
    #TODO: Enter code below for reconstructing the image X_recon_img
    #......................
    #......................
    #X_recon_img = ........
    return X_recon_img

def plot_reconstructions(D,c,num_coeff_array,X_mean,im_num):
    
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,im_num),cmap='Greys_r')
            
    f.savefig('output/hw1b_{0}.png'.format(im_num))
    plt.close(f)
    
    
    
def plot_top_16(D, sz, imname):
    
    f, axarr = plt.subplots(4,4)
    top16=D[:,:16]
    k=0
    for i in range(4):
        for j in range(4):
            plt.axes(axarr[i,j])
            temp=top16[:,k].reshape(sz,sz)
            plt.imshow(temp,cmap='Greys_r')
            k=k+1

    f.savefig(imname)
    plt.close(f)
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    #TODO: Obtain top 16 components of D and plot them
    
    #raise NotImplementedError

    
def main():
    cwd=os.getcwd()
    '''
    Read here all images(grayscale) from Fei_256 folder and collapse 
    each image to get an numpy array Ims with size (no_images, height*width).
    Make sure the images are read after sorting the filenames
    '''
    #TODO: Write a code snippet that performs as indicated in the above comment
    f=[]
    for (dirpath, dirnames, filenames) in walk("Fei_256"):
        f.extend(filenames)
        break
    f=sorted(f[1:], key=natural_key)
    os.chdir("Fei_256")
    Ims=np.zeros([len(f), 256*256])
    for i in range(len(f)):
        im=np.array(Image.open(f[i])).flatten()
        Ims[i,:]=im
    Ims = Ims.astype(np.float32)
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''
    N=16
    D=np.random.rand(N,256*256)
    lmbd=np.zeros(N, dtype=theano.config.floatX)
    eta=0.1

    for i in range(16):
        X_sym=T.matrix('X_sym',dtype=theano.config.floatX)
        d_i=theano.shared(np.random.randn(256*256))
        X_d_i=T.dot(X_sym,d_i)
        if i==0:
            cost=(T.dot(X_d_i.T,X_d_i))*(-1)
        else:
            cost=(T.dot(X_d_i.T,X_d_i)-np.sum(lmbd[j]*T.dot(D[j],d_i)*T.dot(D[j],d_i) for j in range(i)))*(-1)
        gradient=T.grad(cost,d_i)
        Y=d_i-(eta*gradient)
        d_i_update=Y/Y.norm(2)
        f=theano.function([X_sym],updates=[(d_i,d_i_update)])
        t=1
        threshold=0.01
        difference=100
        while t<100 and difference>threshold :
            f(X)
            d_i_new=d_i.get_value()
            difference=np.linalg.norm(D[i]-d_i_new)
            D[i]=d_i_new
            t=t+1
        temp=np.dot(X,D[i])
        lmbd[i]=np.dot(temp.T,temp)

    D=D.T
    c=np.dot(D.T,X.T)
    
    #TODO: Write a code snippet that performs as indicated in the above comment
    os.chdir(cwd)   
    for i in range(0, 200, 10):
        plot_reconstructions(D=D, c=c, num_coeff_array=[1, 2, 4, 6, 8, 10, 12, 14, 16], X_mean=X_mn.reshape((256, 256)), im_num=i)

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')


if __name__ == '__main__':
    main()
    
    