#Oct 31: just checing mean of non-abs corr

if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(PMS)
p_load(oro.nifti)
p_load(neurobase)
p_load(feather)
p_load(glmnet)
p_load(fastBayesReg)
p_load(truncnorm)
p_load(nimble)
p_load(extraDistr)
p_load(rrcov)
p_load(pls)
# install.packages("/well/nichols/users/qcv214/pms2/package/mmand_1.6.2.tar.gz", repos = NULL, type = "source")
library(mmand)


compute_lag1_correlation <- function(img, axis) {
  # Get dimensions of the 3D image
  dims <- dim(img)
  
  # Initialize shifted image
  shifted_img <- array(0, dim = dims)
  
  # Shift the image along the specified axis
  if (axis == "x") {
    shifted_img[2:dims[1], , ] <- img[1:(dims[1]-1), , ]
  } else if (axis == "y") {
    shifted_img[, 2:dims[2], ] <- img[, 1:(dims[2]-1), ]
  } else if (axis == "z") {
    shifted_img[, , 2:dims[3]] <- img[, , 1:(dims[3]-1)]
  }
  
  # Flatten both images to compute correlation
  valid_mask <- shifted_img != 0  # Exclude any potential zero-padded regions
  img_vector <- img[valid_mask]
  shifted_img_vector <- shifted_img[valid_mask]
  
  # Compute correlation between the original and shifted image
  return(cor(img_vector, shifted_img_vector))
}

# Load your 3D nifti image
mask.temp <- oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/viz/oct29_age_cor_fit.nii.gz')
img <- mask.temp@.Data  # Extract the raw data from the nifti object

# Compute lag-1 correlation in each direction (x, y, z)
lag1_corr_x <- compute_lag1_correlation(img, "x")
lag1_corr_y <- compute_lag1_correlation(img, "y")
lag1_corr_z <- compute_lag1_correlation(img, "z")

# Output the correlations
smoothing.cor <- ((lag1_corr_x) + (lag1_corr_y) + (lag1_corr_z))/3
cat("Lag-1 correlation in mean: ", smoothing.cor, "\n")  