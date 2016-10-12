import os
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg

from PIL import Image

import theano
import theano.tensor as T

import re
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def reconstructed_image(D,c,num_coeffs,X_mean,n_blocks,im_num):
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
    
    c_im = c[:num_coeffs,n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
    D_im = D[:,:num_coeffs]
    X1 = np.dot(D_im,c_im)
    sz=int(256/n_blocks)
    X2=X1.T
    X_recon_img=np.zeros([256,256])
    index_u=range(0,257, sz)
    index_v=range(0,257, sz)
    k=0
    for u in range(1,len(index_u)):
        for v in range(1,len(index_v)):
            #temp.shape
            temp=np.array(X2[k,:]).reshape(sz,sz)+X_mean
            #print temp.shape
            X_recon_img[index_u[u-1]:index_u[u],index_v[v-1]:index_v[v]]=temp
            k=k+1

    
    
    

    
    #TODO: Enter code below for reconstructing the image X_recon_img
    #......................
    #......................
    #X_recon_img = ........
    return X_recon_img

def plot_reconstructions(D,c,num_coeff_array,X_mean,n_blocks,im_num):
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
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,n_blocks,im_num),cmap='Greys_r')
            
    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
    plt.close(f)
    
    
    
def plot_top_16(D, sz, imname):
    f, axarr = plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            plt.axes(axarr[i,j])
            temp=np.array(D[:,i*4+j].reshape(sz,sz), dtype=theano.config.floatX)
            plt.imshow(temp,cmap='Greys_r')

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
    cwd = os.getcwd()
    '''
    Read here all images(grayscale) from Fei_256 folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    #TODO: Read all images into a numpy array of size (no_images, height, width)
    f=[]
    for (dirpath, dirnames, filenames) in walk("Fei_256"):
        f.extend(filenames)
        break
    f=sorted(f[1:], key=natural_key)
    os.chdir("Fei_256")
    Ims=np.asarray(np.zeros((len(f),3)), dtype=theano.config.floatX)
    for i in range(len(f)):
        im=Image.open(f[i])
        Ims[i]=i,im.size[0], im.size[1]
    szs = [8, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]

    for sz, nc in zip(szs, num_coeffs):
        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        ''' 
        
        #TODO: Write a code snippet that performs as indicated in the above comment
        X=[]
        for k in range(len(Ims)):
            im=np.array(Image.open('image'+str(int(Ims[k,0]))+'.jpg'),dtype=theano.config.floatX)
            index_u=range(0,257, sz)
            index_v=range(0,257, sz)
            for u in range(1,len(index_u)):
                for v in range(1,len(index_v)):
                    block=np.array(im[index_u[u-1]:index_u[u],index_v[v-1]:index_v[v]]).flatten()
                    X.append(block)
        X=np.array(X, dtype=theano.config.floatX).reshape((int(Ims[0,1])/sz)*200*(int(Ims[0,2])/sz),sz*sz)
        X_mean = np.mean(X, 0)
        X = X - np.repeat(X_mean.reshape(1, -1), X.shape[0], 0)

        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''
        
        #TODO: Write a code snippet that performs as indicated in the above comment
        temp=np.dot(X.T,X)
        lmbda, D=linalg.eig(temp)
        idx = lmbda.argsort()[::-1]   
        lmbda = lmbda[idx]
        D = D[:,idx]
        
        c = np.dot(D.T, X.T)
        cwd1=os.getcwd()
        os.chdir(cwd)
        for i in range(0, 200, 10):
            plot_reconstructions(D=D, c=c, num_coeff_array=nc, X_mean=X_mean.reshape((sz, sz)), n_blocks=int(256/sz), im_num=i)

        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))
        os.chdir(cwd1)


if __name__ == '__main__':
    main()
    
    