#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This source code corresponds to the hyperspectral image denosing method
presented at the 8th Workshop on Hyperspectral Image and Signal
Processing: Evolution in Remote Sensing.

Article:
BBD: A NEW BAYESIAN BI-CLUSTERING DENOISING ALGORITHM FOR
IASI-NG HYPERSPECTRAL IMAGES
https://ieeexplore.ieee.org/servlet/opac?punumber=8053879

(c) 2016, Miguel Colom
http://mcolom.info

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from netCDF4 import Dataset
import argparse
import numpy as np
import numpy.linalg as linalg
from scipy import interpolate
from scipy.cluster.vq import kmeans, vq, whiten
import csv
import os
import sys


def tam(M, name):
    '''
    DEBUG: print out the size of a vector or matrix
    '''
    S = M.shape
    if len(S) == 1:
        print "Vector {} has {} elements".format(name, S[0])
    else:
        print "Matrix {} is {}x{}: {} rows and {} columns".format(name, S[0],S[1], S[0], S[1])

def MSE(I, K):
    '''
    Compute the Mean Squared Error (MSE)
    '''
    return np.mean((I - K)**2.0)

def RMSE(I, K):
    '''
    Compute the square Root of the Mean Squared Error (RMSE)
    '''
    return np.sqrt(MSE(I, K))

def PSNR(I, K):
    '''
    Compute the Peak Signal-to-Noise Ration (PSNR)
    '''
    max_I = np.max(I)
    return 10.0 * np.log10(max_I**2.0 / MSE(I, K))
    
def MSNR_freq(I, K, f):
    '''
    Compute Median Signal-to-Noise Ratio (MSNR)
    '''
    median_I = np.mean(I[f,:])
    return 10.0 * np.log10(median_I**2.0 / MSE(I[f,:], K[f,:]))
    
def print_metadata(dataset):
    '''
    Print out metadata information associated to the granule
    '''
    print "- Source: %s" % dataset.source
    print "- Title: %s" % dataset.title

    print "- Data model: %s" % dataset.data_model
    
    print "- Dimensions:"
    for key in dataset.dimensions.keys():
        d = dataset.dimensions[key]
        print "\t%s;\tsize=%d" % ((d.name, len(d)))
        
    print "- Institution: %s" % dataset.institution
    print "- Orbit: %s" % dataset.orbit


def read_noise_model(filename):
    '''
    Read the CSV file containing the noise model
    '''
    f = open(filename, "r")
    r = csv.reader(f, delimiter=";")
    
    # NEdL, sigma
    data = []
    
    for row in r:
        entry = map(np.float, row)
        data.append(entry)
    
    f.close()

    return data

# Based on:
# http://glowingpython.blogspot.fr/2011/07/principal-component-analysis-with-numpy.html
def princomp(A):
    """ performs principal components analysis 
    (PCA) on the n-by-p data matrix A
    Rows of A correspond to observations, columns to variables. 

    Returns :  
    coeff :
    is a p-by-p matrix, each column containing coefficients 
    for one principal component.
    score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
    latent : 
    a vector containing the eigenvalues 
    of the covariance matrix of A.
    """
    
    '''
    # computing eigenvalues and eigenvectors of covariance matrix
    #tam(A, "A")    
    M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
    #tam(M, "M")
    [latent,coeff] = linalg.eig(np.cov(M)) # attention:not always sorted
    #score = np.dot(coeff.T,M) # projection of the data in the new space
    #return coeff,score,latent
    '''

    # With R
    pid = os.getpid() # To allow parallelization

    filename_csv = "A_{}.csv".format(pid)
    filename_latent = "latent_{}.csv".format(pid)
    filename_coeff = "coeff_{}.csv".format(pid)
    
    np.savetxt(filename_csv, A, delimiter=",")

    # Call R
    os.system("./compute_coeff_latent.R {} {} {}".
      format(filename_csv, filename_latent, filename_coeff))

    # Read generated coeff and latent data
    coeff = np.load(filename_coeff)
    latent = np.load(filename_latent)
    
    # Clean up
    os.remove(filename_csv)
    os.remove(filename_latent)
    os.remove(filename_coeff)

    return coeff, latent

def pca_unproject(W, S, original_data_add_mean):
    '''
    Projects back a signal after PCA transformation
    '''
    M = np.dot(W.T, S) # If W was square,
                       # this would be M = np.dot(W_inv, S)
    #M = np.dot(np.linalg.pinv(W), S) # With the pseudoinverse
    A = M.T + np.mean(original_data_add_mean, axis=1) # Add the mean
    return A.T

def get_interpolation_function(csv_filename):
    '''
    Return an interpolation function according to the noise model in
    the CSV files
    '''
    noise_model = read_noise_model(csv_filename) # NEdL, nu
    
    nedl = np.zeros(len(noise_model))
    sigma_nedl = np.zeros(len(noise_model))

    for i in range(len(noise_model)):
        nedl[i] = noise_model[i][0]
        sigma_nedl[i] = noise_model[i][1]
    
    return interpolate.interp1d(nedl, sigma_nedl)

def print_evaluation(S_name, N, denoised, band, band_noisy):   
    print "N={}, using {}".format(N, S_name)

    MSE_noisy_ref = MSE(band_noisy, band[:,:])
    MSE_denoised_ref = MSE(denoised, band[:,:])
    PSNR_ref_noisy = PSNR(band[:,:], band_noisy)
    PSNR_ref_denoised = PSNR(band[:,:], denoised)
    
    print "MSE(band_noisy, band)={}".format(MSE_noisy_ref)
    print "MSE(denoised, band)={}".format(MSE_denoised_ref)
    print "PSNR(band, band_noisy)={}".format(PSNR_ref_noisy)
    print "PSNR(band, denoised)={}".format(PSNR_ref_denoised)
    print "Ratio MSE: {}".format(MSE_noisy_ref / MSE_denoised_ref)
    print "Gain PSNR: {}".format(PSNR_ref_denoised - PSNR_ref_noisy)

def read_granule_all_bands(dataset):
    '''
    Reads a granule given its dataset and returns the 3D data and the
    associated wavenumbers
    '''
    num_bands = 0
    while 'spectrum_band%d' % (num_bands+1) in dataset.variables.keys():
        num_bands += 1
            
    # Get the number of frequencies
    num_freqs = 0
    for band_num in range(1,num_bands+1):
        band = dataset.variables["spectrum_band%d" % band_num][:-1, :] # spectrum band, without last freq (duplicated)
        num_freqs += band.shape[0]

    # Read all bands
    granule = np.zeros((num_freqs, band.shape[1]))
    freqs = np.zeros(num_freqs)
    
    f = 0
    for band_num in range(1,num_bands+1):
        band = dataset.variables["spectrum_band%d" % band_num][:-1, :] # spectrum band, without last freq (duplicated)
        freqs_band = dataset.variables['wavenumber_band%d' % band_num][:-1] # without last (duplicated)
        num_freqs_band = band.shape[0]
        granule[f:f+num_freqs_band, :] = band[:,:]
        freqs[f:f+num_freqs_band] = freqs_band

        f += num_freqs_band
    
    return granule, freqs

def get_coeff_latent_filenames(granule_num, label, Q):
    '''
    Reads the coefficients and latent in the temporary file which
    R computed
    '''
    coeff_filename  = "gi_precomp/coeff_granule{}_Q{}_label{}.npy".format(granule_num, Q, label)
    latent_filename = "gi_precomp/latent_granule{}_Q{}_label{}.npy".format(granule_num, Q, label)
        
    return coeff_filename, latent_filename

def store_indexed(local_from, global_to, indices_loc2global_dict):
    '''
    Stores data in a 3D cube from local (inside a cluster) to global
    (inside the cube)
    '''
    for lf in range(local_from.shape[0]):
        gf = indices_loc2global_dict[lf]
        global_to[gf, :] += local_from[lf, :]

def ApplyBitTrimming(Spectre,RequiredBits,SpectreMin,SpectreMax):
    '''
    Applies bit-trimming compressing to the given granule
    '''
    deux = np.zeros(RequiredBits.shape)+2
    K = (np.power(deux,RequiredBits)-1)/(SpectreMax - SpectreMin)
    Spectre_trim = np.round(K*(Spectre-SpectreMin))
    Spectre_trim.astype(int)
    return Spectre_trim
    
def ReverseBitTrimming(Spectre_trim,RequiredBits,SpectreMin,SpectreMax):
    '''
    Reverses bit-trimming to obtain a de-trimmed spectrum
    '''
    deux = np.zeros(RequiredBits.shape)+2
    K = (np.power(deux,RequiredBits)-1)/(SpectreMax - SpectreMin)
    Spectre_detrim = Spectre_trim.astype(float)/K+SpectreMin
    return Spectre_detrim    
    

# Parse arguments
parser = argparse.ArgumentParser(description="Hyperspectral imaging denoising (BBD) method. (c) 2016 Miguel Colom. http://mcolom.info")
parser.add_argument("granule_num", help='Number of the granule')
parser.add_argument("Q", help='Number of clusters')
parser.add_argument("-a", "-avoid",
                    action='store_const', const=True, default=False,
                    help='Avoid denoising. Only clustering and PCA')
parser.parse_args()
args = parser.parse_args()

# Read arguments
granule_num = int(args.granule_num)
Q = int(args.Q) # Number of spectral clusters
do_denoising = not args.a
apply_bit_trimming = False

# Create memoization directory if it doesn't exist
memo_dir = os.path.dirname("gi_precomp/")
try:
    os.stat(memo_dir)
except:
    os.mkdir(memo_dir)

# Read granule
filename_nc = 'granule{}.nc'.format(granule_num)
dataset = Dataset(filename_nc, 'r')
print_metadata(dataset)

# Get number of bands
num_bands = 0
while 'spectrum_band%d' % (num_bands+1) in dataset.variables.keys():
    num_bands += 1

# Check if wavenumbers are sorted
for i in range(1,num_bands+1):
    assert(np.alltrue(np.sort(dataset.variables["wavenumber_band%d" % i][:]) ==  np.sort(dataset.variables["wavenumber_band%d" % i][:])))

granule, freqs = read_granule_all_bands(dataset)
    
# Add noise
granule_noisy = np.zeros(granule.shape)

function_inter_nedl = get_interpolation_function("Bruit.csv")
#
filename = "gi_precomp/granule{}_noisy.npy".format(granule_num)
if not os.path.isfile(filename):
    np.random.seed(1234) # Deterministic noise, for comparison sake
    for i in range(granule.shape[0]):
        sigma = function_inter_nedl(freqs[i])
        granule_noisy[i,:] = granule[i,:] + np.random.normal(loc=0.0, scale=sigma, size=granule.shape[1])
    np.save(filename, granule_noisy)
    
    filename_freqs = "gi_precomp/granule{}_freqs.npy".format(granule_num)
    np.save(filename_freqs, freqs)
else:
    granule_noisy = np.load(filename)
    
### Apply bit-trimming
if apply_bit_trimming:
    # Min and max of the spectrum for all pixels
    spectre_min= np.amin(granule_noisy, axis=1)
    spectre_max= np.amax(granule_noisy, axis=1)

    # Get number of bits at each wavenumber
    ratio = (spectre_max - spectre_min) / function_inter_nedl(freqs)
    required_bits = np.trunc(np.log(ratio)/np.log(2))+1

    # Apply bit-trimming
    granule_noisy_bitTrimming = np.zeros(granule_noisy.shape)
    #
    for i in range(granule_noisy.shape[1]): # Do BT at each pixel
        spectre_trimmed = ApplyBitTrimming(granule_noisy[:,i], required_bits, spectre_min, spectre_max)
        granule_noisy_bitTrimming[:,i] = ReverseBitTrimming(spectre_trimmed, required_bits, spectre_min, spectre_max)

    # Compute quantization error
    sigma2_bitTrimming = np.zeros(granule_noisy.shape[0])
    for i in range(sigma2_bitTrimming.shape[0]):
        sigma2_bitTrimming[i] = np.var(np.abs(granule_noisy_bitTrimming[i,:] - granule_noisy[i,:]), ddof=1)


######################################################################
#                           BBD method                               #
######################################################################

N = 20  # Global number of PCs kept
K = 400 # Number of similar pixels, excluding the reference

# Clustering
filename = "gi_precomp/code_granule{}_Q{}.npy".format(granule_num, Q)
#    
if not os.path.isfile(filename):
    data_whiten = whiten(granule_noisy_bitTrimming if apply_bit_trimming else granule_noisy)
    codebook, distortion = kmeans(data_whiten, Q)
    code, _ = vq(data_whiten, codebook)
    np.save(filename, code)
else:
    code = np.load(filename)

# The final results are written here
granule_denoised = np.zeros(granule.shape)

# Process each spectral cluster
for label in range(Q):
    print "Label: {}".format(label)

    F = len(code[code == label]) # Number of frequencies

    # Noisy band, band with noise and frequencies, for those frequencies
    # in the cluster with that label
    band_noisy = np.zeros((F, granule.shape[1]))
    band = np.zeros((F, granule.shape[1]))
    freqs_band = np.zeros(F)

    # Store indexed
    count = 0
    indices_loc2global_dict = {}
    for f in range(granule.shape[0]):
        if code[f] == label:
            band_noisy[count,:] = granule_noisy_bitTrimming[f,:] if apply_bit_trimming else granule_noisy[f,:]
            band[count,:] = granule[f,:]
            freqs_band[count] = freqs[f]
            indices_loc2global_dict[count] = f
            count += 1
    assert(count == F)

    # Do PCA of noisy band
    print "PCA label {} begins".format(label)
    coeff, latent = princomp(band_noisy[:,:])
    print "PCA label {} ends".format(label)
    
    # Descending sorting indices for the absolute values of the eigenvalues
    idx = np.argsort(np.abs(latent))[::-1]
    
    # Project band_noisy over the N most significant PCs
    W = coeff.T
    W = W[idx[0:N], :]  # The first N rows of the matrix are chosen
    W = np.real(W)

    # Subtract mean
    M = (band_noisy[:,:].T-np.mean(band_noisy[:,:],axis=1)).T

    S = np.dot(W, M) # Projection
    S = np.real(S)

    # Compute the covariance matrix of the noise
    D = np.zeros((band_noisy.shape[0], band_noisy.shape[0]))
    for i in range(band_noisy.shape[0]):
        sigma = function_inter_nedl(freqs_band[i])

        sigma2_final = sigma**2.0
        if apply_bit_trimming:
            sigma2_final += sigma2_bitTrimming[indices_loc2global_dict[i]]
        #
        D[i,i] = sigma2_final # Add extra variance because of the BT

    Cn = np.dot(W, D).dot(W.T)
    Cn = np.real(Cn)

    # Correlation matrix, to choose the most similar pixels
    Rp = np.corrcoef(S.T)

    ### Denoising
    S_denoised = np.zeros(S.shape)
        
    for pixel_index in range(S.shape[1]):
        #print pixel_index
        
        Pn = S[:, pixel_index] # Noisy pixel
        
        if do_denoising:
            # Denoising Pn...
            sim_idx = np.argsort(Rp[pixel_index])[::-1] # Indices of the most similar pixels
            
            # Average of the K+1 similar pixels, including Pn itself
            Pn_mean = np.mean(S[:, sim_idx[0:K+1]], axis=1)
            
            # Covariance matrix of the similar pixels
            Cpn = np.cov(S[:, sim_idx[1:K+1]])
            Cpn_inv = np.linalg.inv(Cpn)

            # Denoising
            P = Pn_mean + np.dot(Cpn - Cn, Cpn_inv).dot(Pn - Pn_mean)
            S_denoised[:, pixel_index] = P
        else:
            # Test: do not denoise. Only clustering and PCA
            S_denoised[:, pixel_index] = Pn

    ### Project back
    #
    # With denoised (S_denoised)
    denoised = pca_unproject(W, S_denoised, band_noisy[:,:])
    #
    store_indexed(denoised, granule_denoised, indices_loc2global_dict)

    print


# Save denoised granule
if do_denoising:
    filename = "gi_precomp/granule{}_denoised_Q{}.npy".format(granule_num, Q)
else:
    filename = "gi_precomp/granule{}_NoDenoised_Q{}.npy".format(granule_num, Q)

# Compute a pondered mean in order that the variance of the removed signal coincides with that of the noise model
# D --> alpha*D + (1-alpha) * S
#   D: denoised
#   S: noisy
noise   = granule_noisy - granule
removed = granule_noisy - granule_denoised

for i in range(granule.shape[0]):
    alpha2 = np.var(noise[i,:], ddof=1) / (np.var(granule_noisy[i,:], ddof=1)  + np.var(granule_denoised[i,:], ddof=1) - 2*np.cov(granule_noisy[i,:], granule_denoised[i,:])[0,1])
    assert(alpha2 >= 0)
    alpha = np.sqrt(alpha2)
    #
    granule_denoised[i,:] = alpha*granule_denoised[i,:] + (1.0-alpha)*granule_noisy[i,:]

np.save(filename, granule_denoised)
