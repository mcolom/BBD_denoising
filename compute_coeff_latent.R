#!/usr/bin/env Rscript

# This source code corresponds to the hyperspectral image denosing method
# presented at the 8th Workshop on Hyperspectral Image and Signal
# Processing: Evolution in Remote Sensing.
#
# Article:
# BBD: A NEW BAYESIAN BI-CLUSTERING DENOISING ALGORITHM FOR
# IASI-NG HYPERSPECTRAL IMAGES
#
# (c) 2016, Miguel Colom
# http://mcolom.info

# This file computes the PCA using R

library(RcppCNPy)

args = commandArgs(trailingOnly=TRUE)
if (length(args) != 3) {
  stop("You need to give the file name of the matrix!")
}

# Get file names
filename_csv = args[1]
filename_latent = args[2]
filename_coeff = args[3]

# Load the CSV file
input = read.csv(filename_csv, header=FALSE)

# Compute PCA
prp <- prcomp(t(input))
eigp <- (prp$sdev)^2 # These eigp are the "latent" in Python

# Save latent
RcppCNPy::npySave(filename_latent, eigp)

# prp$rotation is matrix coeff in Python

# Save coeff
RcppCNPy::npySave(filename_coeff, prp$rotation)
