#29 Aug: modelling pm_tf

#sep20 including co.dat with caterogrical
#sep24 inclde co.dat with continuous var

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


##########
##########
print("load data")
##Get pm response from kgpnn project

age_tab <-  as.data.frame(read_feather('/well/nichols/users/qcv214/KGPNN/cog/agesex_strat2.feather'))

train.test.ind <- list()
train.test.ind$test <- read.csv('/well/nichols/users/qcv214/KGPNN/cog/cog_test_index.csv')$x
train.test.ind$train <-  read.csv('/well/nichols/users/qcv214/KGPNN/cog/cog_train_index.csv')$x
n.train <- length(train.test.ind$train)

# age_tab.test <- age_tab[101:200,]
# age_tab <- age_tab[1:100,]


#########
age <- as.numeric(age_tab$age)
sex <-  as.numeric(age_tab$sex)
sex <- sapply(sex, function(x) replace(x, x==0,-1)) #Change female to -1, male to 1
depind <- age_tab$DepInd
co.dat <- cbind(sex,depind,age)
########

#Over_writing age with pm_tf so I dont have to change the subsequent code
age_tab$age <- age_tab$pm_tf

age_tab.test <- age_tab[train.test.ind$test,]
age_tab <- age_tab[train.test.ind$train,]

##Get data 

list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
# sub.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
sub.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz'))
print("here 1")

column.names <- as.character(1:ncol(sub.dat))
print("here 2")

colnames(sub.dat) <- as.character(1:ncol(sub.dat))
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab.test$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
# sub.dat.test <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
sub.dat.test <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz'))
print("here 3")

colnames(sub.dat.test) <- colnames(sub.dat)

print("here 4")
####Calling mmand
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')

colnames(sub.dat) <- column.names
print("here 8")

colnames(sub.dat.test) <- column.names

##############################################################################################################################
######Non-smooth
num.vox.vec <- (1:25)*400

print("ridge")
#Ridge

# set.seed(4)
# fit.ridge <- glmnet(sub.dat,age_tab$age, alpha=0,lambda=1e-4)
# beta <- coef(fit.ridge)
# beta_no_int <- beta[-1,]
# rank_beta.ridge <- beta_no_int[order(abs(beta_no_int), decreasing=TRUE)]
# 
# long.rmse <- function(ranked_coef, num.vox.vec){
#   train <- vector(mode = 'numeric')
#   test <- vector(mode = 'numeric')
#   for(i in num.vox.vec){
#     var.sel<- names(ranked_coef[1:i])
#     # fit.ridge.pca <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0)
#     fit.ridge.pca <- cv.glmnet(cbind(sub.dat[,var.sel],co.dat[train.test.ind$train, ]),age_tab$age, alpha=0)  #fixing lambda
#     beta <- coef(fit.ridge.pca, s = "lambda.min")
#     train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat[,var.sel],co.dat[train.test.ind$train, ])%*%beta))^2)))
#     test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test[,var.sel],co.dat[train.test.ind$test, ])%*%beta))^2)))
#   }
#   out <- list()
#   out$train <- train
#   out$test <- test
#   return(out)
# }
# 
# res.ridge <- long.rmse(rank_beta.ridge, num.vox.vec)
# #Full Ridge
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0] <- abs(c(beta_no_int))
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_ridge'))
# 
# var.sel <- (order(abs(beta_no_int), decreasing=TRUE))
# print(length(c(var.sel)))
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_ridge_rank'))


theta.range <- c(1e-4,1e-3,1e-1,1,1e3)


#Robust PCA

# 
# simple.cv <- function(lambda, no.variable = 5000,theta.range. = theta.range){ #Changed from -9 to -8, to -7 for 2k
#   train <- vector(mode = 'numeric')
#   test <- vector(mode = 'numeric')
#   for(theta in theta.range.){
#     #doing PCA
#     # print(1)
#     Omeg <- solve(sub.dat%*%lambda%*%t(sub.dat) + (theta)*diag(nrow(sub.dat))) #4x4 
#     # print(2)
#     pre_beta_pms <- lambda%*%t(sub.dat)%*%Omeg #4 x 75
#     # print(3)
#     beta_pms <- pre_beta_pms%*%age_tab$age
#     # print(4)
#     rownames(beta_pms) <- colnames(sub.dat)
#     rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
#     
#     #Doing prediction
#     var.sel<- names(rank_beta_pms[1:no.variable])
#     # print(5)
#     fit.ridge <- cv.glmnet(cbind(sub.dat[,var.sel],co.dat[train.test.ind$train, ]),age_tab$age, alpha=0) 
#     beta <- coef(fit.ridge, s = "lambda.min")
#     train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat[,var.sel],co.dat[train.test.ind$train, ])%*%beta))^2)))
#     test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test[,var.sel],co.dat[train.test.ind$test, ])%*%beta))^2)))
#   }
#   out <- rbind(theta.range,train,test)
#   return(out)
# }
# 
# #below is practically long.rmse
# print("making predictions")
# long.rmse.with.lambda.theta.ridge <- function(lambda,theta,num.vox.vec){
#   Omeg <- solve(sub.dat%*%lambda%*%t(sub.dat) + (theta)*diag(nrow(sub.dat))) #4x4 
#   pre_beta_pms <- lambda%*%t(sub.dat)%*%Omeg #4 x 75
#   beta_pms <- pre_beta_pms%*%age_tab$age
#   rownames(beta_pms) <- colnames(sub.dat)
#   rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
#   train <- vector(mode = 'numeric')
#   test <- vector(mode = 'numeric')
#   for(i in num.vox.vec){
#     print(i)
#     var.sel<- names(rank_beta_pms[1:i])
#     #ridge prediction
#     # fit.ridge <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0)
#     fit.ridge <- cv.glmnet(cbind(sub.dat[,var.sel],co.dat[train.test.ind$train, ]),age_tab$age, alpha=0) #fixing lambda
#     # beta <- coef(fit.ridge, s = "lambda.min")
#     beta <- coef(fit.ridge, s = "lambda.min")
#     train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat[,var.sel],co.dat[train.test.ind$train, ])%*%beta))^2)))
#     test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test[,var.sel],co.dat[train.test.ind$test, ])%*%beta))^2)))
#   }
#   out <- list()
#   out$train <- train
#   out$test <- test
#   return(out)
# }
# 
# 
# # pca <- PcaHubert(sub.dat,kmax=100) 
# # print("rmse robpca")
# print("tune pca")
# pca <-  prcomp(sub.dat)
# explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
# # Calculate the cumulative variance explained
# cumulative_variance <- cumsum(explained_variance)
# # Determine the number of components needed to explain at least 90% of the variance
# num_components.90 <- which(cumulative_variance >= 0.90)[1]
# 
# 
# # print("tune pca 90")
# lambda.80 <- pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90])
# pca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
# print("rmse pca 90")
# res.lambda.pca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, pca.cv.90[1,which.min(pca.cv.90[3,])], num.vox.vec)
# 
# Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (pca.cv.90[1,which.min(pca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
# pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
# beta_pms <- pre_beta_pms%*%age_tab$age
# rownames(beta_pms) <- colnames(sub.dat)
# rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
# 
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0] <- abs(c(beta_pms))
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_PCA'))
# 
# ###
# var.sel <- (order(abs(beta_pms), decreasing=TRUE))
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_PCA_rank'))
# 
# ####Removing previous data
# remove(lambda.80)
# remove(pre_beta_pms)
# remove(mask.temp)
# remove(beta_pms)
# remove(Omeg)
# 
# 
# print("tune SPCA")
# 
# cv_model <- cv.glmnet(sub.dat, age_tab$age, alpha = 0, nfolds = 10)
# # Best lambda from cross-validation
# coefficients <- coef(cv_model, s = "lambda.min")  # Extract coefficients at best lambda
# weights <- abs(coefficients[-1,])  # Exclude intercept from weights (first row of coefficients)
# X_weighted <- scale(sub.dat) * sqrt(weights)
# # Perform PCA on the weighted matrix
# pca<- prcomp(X_weighted, center = TRUE, scale. = FALSE)
# 
# explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
# # Calculate the cumulative variance explained
# cumulative_variance <- cumsum(explained_variance)
# # Determine the number of components needed to explain at least 90% of the variance
# num_components.90 <- which(cumulative_variance >= 0.90)[1]
# 
# print("tune pca 90")
# lambda.80 <- pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90])
# spca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
# print("rmse SPCA 90")
# res.lambda.spca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, spca.cv.90[1,which.min(spca.cv.90[3,])], num.vox.vec)
# 
# Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (pca.cv.90[1,which.min(pca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
# pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
# beta_pms <- pre_beta_pms%*%age_tab$age
# rownames(beta_pms) <- colnames(sub.dat)
# rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
# 
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0] <- abs(c(beta_pms))
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_SPCA'))
# 
# ###
# var.sel <- (order(abs(beta_pms), decreasing=TRUE))
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_SPCA_rank'))
# 
# ####Removing previous data
# remove(lambda.80)
# remove(pre_beta_pms)
# remove(mask.temp)
# remove(beta_pms)
# remove(Omeg)
# remove(pca)
# remove(lambda.80)
# remove(weights)
# remove(X_weighted)
# 
# 
# out <- rbind(res.lambda.pca.90$train,res.lambda.spca.90$train ,res.ridge$train, 
#              res.lambda.pca.90$test,res.lambda.spca.90$test,res.ridge$test)
# write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/oct29_pm_pms_full.csv'), row.names = FALSE)


beta_pms <- c(apply(sub.dat, 2, function(voxel_data) cor(voxel_data, age_tab$age)))
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0] <- beta_pms
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_cor_fit'))

remove(mask.temp)
remove(beta_pms)

#########Compute voxel-wise corr for smoothing
##Look at voxel-wise corr
# Function to compute lag-1 correlation along a specific axis (x, y, or z)
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
mask.temp <- oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/viz/oct29_pm_cor_fit.nii.gz')
img <- mask.temp@.Data  # Extract the raw data from the nifti object

# Compute lag-1 correlation in each direction (x, y, z)
lag1_corr_x <- compute_lag1_correlation(img, "x")
lag1_corr_y <- compute_lag1_correlation(img, "y")
lag1_corr_z <- compute_lag1_correlation(img, "z")

# Output the correlations
smoothing.cor <- (abs(lag1_corr_x) + abs(lag1_corr_y) + abs(lag1_corr_z))/3
cat("Lag-1 correlation in mean: ", smoothing.cor, "\n")  


##################################################################################################################################
#################### PMS and HOLP
# 
# #PMS
# img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id[1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
# nb <- find_brain_image_neighbors(img1, res3.mask, radius=1)
# 
# n.train <- nrow(sub.dat)
# n.test <- nrow(sub.dat.test)
# rho_high <- rep(-log(smoothing.cor)/4,length=ncol(sub.dat))
# 
# print("stage 3")
# 
# #pms high
# out.pms.hi<-fast_PMS_local_spatial(x=as.matrix(sub.dat), y = age_tab$age, coords=nb$maskcoords,neighbors=nb$mask_img_nb,num_neighbors=nb$num_neighbors, rho = rho_high)$pms_select
# #holp
# out.holp<-fast_PMS_cpp(x=as.matrix(sub.dat), y = age_tab$age, theta = 0)$pms_select
# 
# 
# long.rmse.pms <- function(pms_ind,num.vox.vec){ #this function is extremely similar to long.rmse for ridge, but without "names( ind )"
#   train <- vector(mode = 'numeric')
#   test <- vector(mode = 'numeric')
#   for(i in num.vox.vec){
#     print(i)
#     var.sel<- pms_ind[1:i]
#     #ridge prediction
#     # fit.ridge <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0)
#     fit.ridge <- cv.glmnet(cbind(sub.dat[,var.sel],co.dat[train.test.ind$train, ]),age_tab$age, alpha=0) #fixing lambda
#     # beta <- coef(fit.ridge, s = "lambda.min")
#     beta <- coef(fit.ridge, s = "lambda.min")
#     train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat[,var.sel],co.dat[train.test.ind$train, ])%*%beta))^2)))
#     test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test[,var.sel],co.dat[train.test.ind$test, ])%*%beta))^2)))
#   }
#   out <- list()
#   out$train <- train
#   out$test <- test
#   return(out)
# }
# 
# res.pms.hi <- long.rmse.pms(out.pms.hi, num.vox.vec)
# res.pms.holp <- long.rmse.pms(out.holp, num.vox.vec)
# 
# out <- rbind(res.pms.hi$train,res.pms.holp$train, 
#              res.pms.hi$test,res.pms.holp$test)
# 
# write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/oct29_pm_nbpms_full.csv'), row.names = FALSE)
# 
# #plot out.pms.hi
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0][out.pms.hi] <- c(ncol(sub.dat):1)/ncol(sub.dat)
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_pms_hi'))
# 
# #plot out.pms.holp
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0][out.holp] <- c(ncol(sub.dat):1)/ncol(sub.dat)
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_pms_holp'))


########################################################################################################################################################
################################################################ Smoothing #############################################################################

smoothing.sigma <-  1/sqrt(-2*log(smoothing.cor))
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
smooth_1d_image <- function(x){
  #turn 1d image into 3d
  gp.mask.hs <- res3.mask
  gp.mask.hs[gp.mask.hs!=0] <- x
  #smoothen
  gp.mask.hs <- gaussianSmooth(gp.mask.hs, c(smoothing.sigma,smoothing.sigma,smoothing.sigma))
  #mask it again
  gp.mask.hs <- gp.mask.hs[res3.mask!=0]
  return(gp.mask.hs)
}

sub.dat <- t(apply(sub.dat,MARGIN = 1, smooth_1d_image))
print(dim(sub.dat))
sub.dat.test <- t(apply(sub.dat.test,MARGIN = 1, smooth_1d_image))
colnames(sub.dat) <- column.names
colnames(sub.dat.test) <- column.names

print("ridge")
#Ridge

set.seed(4)
fit.ridge <- glmnet(sub.dat,age_tab$age, alpha=0, lambda=1e-4)
beta <- coef(fit.ridge)
beta_no_int <- beta[-1,]
rank_beta.ridge <- beta_no_int[order(abs(beta_no_int), decreasing=TRUE)]

long.rmse <- function(ranked_coef, num.vox.vec){
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(i in num.vox.vec){
    var.sel<- names(ranked_coef[1:i])
    # fit.ridge.pca <- cv.glmnet(sub.dat.s[,var.sel],age_tab$age, alpha=0)
    fit.ridge.pca <- cv.glmnet(cbind(sub.dat[,var.sel],co.dat[train.test.ind$train, ]),age_tab$age, alpha=0) #fixing lambda
    beta <- coef(fit.ridge.pca, s = "lambda.min")
    train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat[,var.sel],co.dat[train.test.ind$train, ])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test[,var.sel],co.dat[train.test.ind$test, ])%*%beta))^2)))
  }
  out <- list()
  out$train <- train
  out$test <- test
  return(out)
}

res.ridge <- long.rmse(rank_beta.ridge, num.vox.vec)

#Full Ridge
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_no_int))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_ridge'))

var.sel <- (order(abs(beta_no_int), decreasing=TRUE))
print(length(c(var.sel)))
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_ridge_rank'))


simple.cv <- function(lambda, no.variable = 5000,theta.range. = theta.range){ #Changed from -9 to -8, to -7 for 2k
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(theta in theta.range.){
    #doing PCA
    Omeg <- solve(sub.dat%*%lambda%*%t(sub.dat) + (theta)*diag(nrow(sub.dat))) #4x4 
    pre_beta_pms <- lambda%*%t(sub.dat)%*%Omeg #4 x 75
    beta_pms <- pre_beta_pms%*%age_tab$age
    rownames(beta_pms) <- colnames(sub.dat)
    rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
    
    #Doing prediction
    var.sel<- names(rank_beta_pms[1:no.variable])
    fit.ridge <- cv.glmnet(cbind(sub.dat[,var.sel],co.dat[train.test.ind$train, ]),age_tab$age, alpha=0)
    beta <- coef(fit.ridge, s = "lambda.min")
    train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat[,var.sel],co.dat[train.test.ind$train, ])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test[,var.sel],co.dat[train.test.ind$test, ])%*%beta))^2)))
  }
  out <- rbind(theta.range,train,test)
  return(out)
}


#below is practically long.rmse
print("making predictions")
long.rmse.with.lambda.theta.ridge <- function(lambda,theta,num.vox.vec){
  Omeg <- solve(sub.dat%*%lambda%*%t(sub.dat) + (theta)*diag(nrow(sub.dat))) #4x4 
  pre_beta_pms <- lambda%*%t(sub.dat)%*%Omeg #4 x 75
  beta_pms <- pre_beta_pms%*%age_tab$age
  rownames(beta_pms) <- colnames(sub.dat)
  rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(i in num.vox.vec){
    # print(i)
    var.sel<- names(rank_beta_pms[1:i])
    #ridge prediction
    # fit.ridge <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0)
    fit.ridge <- cv.glmnet(cbind(sub.dat[,var.sel],co.dat[train.test.ind$train, ]),age_tab$age, alpha=0) #fixing lambda
    # beta <- coef(fit.ridge, s = "lambda.min")
    beta <- coef(fit.ridge, s = "lambda.min")
    train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat[,var.sel],co.dat[train.test.ind$train, ])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test[,var.sel],co.dat[train.test.ind$test, ])%*%beta))^2)))
  }
  out <- list()
  out$train <- train
  out$test <- test
  return(out)
}

print("tune pca")
pca <-  prcomp(sub.dat)
explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
# Calculate the cumulative variance explained
cumulative_variance <- cumsum(explained_variance)
# Determine the number of components needed to explain at least 90% of the variance
num_components.90 <- which(cumulative_variance >= 0.90)[1]

################################################## No shrinkage
print("tune pca 90")
lambda.80 <- (pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90]))
# diag(lambda.80) <- diag(lambda.80) + (1-shrinkage.param)
####
pca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
print("rmse pca 90")
res.lambda.pca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, pca.cv.90[1,which.min(pca.cv.90[3,])], num.vox.vec)

#Plotting 
Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (pca.cv.90[1,which.min(pca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
beta_pms <- pre_beta_pms%*%age_tab$age
rownames(beta_pms) <- colnames(sub.dat)
rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_pms))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_PCA'))

var.sel <- (order(abs(beta_pms), decreasing=TRUE))
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_PCA_rank'))

####Removing previous data
remove(lambda.80)
remove(pre_beta_pms)
remove(mask.temp)
remove(beta_pms)
remove(Omeg)
##################################################
print("tune pca 90")

shrinkage.param <- 0.3

lambda.80 <- shrinkage.param*(pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90]))
diag(lambda.80) <- diag(lambda.80) + (1-shrinkage.param)
####
pca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
print("rmse pca 90")
res.lambda.pca.90.p <- long.rmse.with.lambda.theta.ridge(lambda.80, pca.cv.90[1,which.min(pca.cv.90[3,])], num.vox.vec)

#Plotting 
Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (pca.cv.90[1,which.min(pca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
beta_pms <- pre_beta_pms%*%age_tab$age
rownames(beta_pms) <- colnames(sub.dat)
rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_pms))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_PCA_proj'))
var.sel <- (order(abs(beta_pms), decreasing=TRUE))
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_PCA_proj_rank'))

####Removing previous data
remove(lambda.80)
remove(pre_beta_pms)
remove(mask.temp)
remove(beta_pms)
remove(Omeg)
###### Cov
lambda.80 <- shrinkage.param*pca$rotation[, 1:num_components.90] %*% diag(pca$sdev[1:num_components.90]^2) %*% t(pca$rotation[, 1:num_components.90])
diag(lambda.80) <- diag(lambda.80) + (1-shrinkage.param)
####
pca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
print("rmse pca 90")
res.lambda.pca.90.c <- long.rmse.with.lambda.theta.ridge(lambda.80, pca.cv.90[1,which.min(pca.cv.90[3,])], num.vox.vec)

#Plotting 
Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (pca.cv.90[1,which.min(pca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
beta_pms <- pre_beta_pms%*%age_tab$age
rownames(beta_pms) <- colnames(sub.dat)
rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_pms))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_PCA_cov'))
var.sel <- (order(abs(beta_pms), decreasing=TRUE))
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_PCA_cov_rank'))
########################################
####Removing previous data
remove(lambda.80)
remove(pre_beta_pms)
remove(mask.temp)
remove(beta_pms)
remove(Omeg)
###### Cov

################################################################################ 

print("tune SPCA")

# cv_model <- glmnet(sub.dat, age_tab$age, alpha = 0, lambda=1e-4)
# # Best lambda from cross-validation
# coefficients <- coef(cv_model)  # Extract coefficients at best lambda
# weights <- abs(coefficients[-1,])  # Exclude intercept from weights (first row of coefficients)
# X_weighted <- scale(sub.dat) * sqrt(weights)
# # Perform PCA on the weighted matrix
# pca<- prcomp(X_weighted, center = TRUE, scale. = FALSE)
# 
# explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
# # Calculate the cumulative variance explained
# cumulative_variance <- cumsum(explained_variance)
# # Determine the number of components needed to explain at least 90% of the variance
# num_components.90 <- which(cumulative_variance >= 0.90)[1]
# 
# 
# ################################################## No shrinkage
# lambda.80 <- (pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90]))
# 
# spca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
# print("rmse SPCA 90")
# res.lambda.spca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, spca.cv.90[1,which.min(spca.cv.90[3,])], num.vox.vec)
# 
# #Plotting 
# Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (spca.cv.90[1,which.min(spca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
# pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
# beta_pms <- pre_beta_pms%*%age_tab$age
# rownames(beta_pms) <- colnames(sub.dat)
# rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
# 
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0] <- abs(c(beta_pms))
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_SPCA'))
# var.sel <- (order(abs(beta_pms), decreasing=TRUE))
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_SPCA_rank'))
# 
# ####Removing previous data
# remove(lambda.80)
# remove(pre_beta_pms)
# remove(mask.temp)
# remove(beta_pms)
# remove(Omeg)
# ################################################## 
# 
# 
# 
# print("tune pca 90")
# lambda.80 <- shrinkage.param*(pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90]))
# diag(lambda.80) <- diag(lambda.80) + (1-shrinkage.param)
# 
# spca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
# print("rmse SPCA 90")
# res.lambda.spca.90.p <- long.rmse.with.lambda.theta.ridge(lambda.80, spca.cv.90[1,which.min(spca.cv.90[3,])], num.vox.vec)
# 
# #Plotting 
# Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (spca.cv.90[1,which.min(spca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
# pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
# beta_pms <- pre_beta_pms%*%age_tab$age
# rownames(beta_pms) <- colnames(sub.dat)
# rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
# 
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0] <- abs(c(beta_pms))
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_SPCA_proj'))
# var.sel <- (order(abs(beta_pms), decreasing=TRUE))
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_SPCA_proj_rank'))
# 
# ####Removing previous data
# remove(lambda.80)
# remove(pre_beta_pms)
# remove(mask.temp)
# remove(beta_pms)
# remove(Omeg)
# ###### Cov
# ###### Cov
# lambda.80 <- shrinkage.param*pca$rotation[, 1:num_components.90] %*% diag(pca$sdev[1:num_components.90]^2) %*% t(pca$rotation[, 1:num_components.90])
# diag(lambda.80) <- diag(lambda.80) + (1-shrinkage.param)
# 
# spca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
# print("rmse SPCA 90")
# res.lambda.spca.90.c <- long.rmse.with.lambda.theta.ridge(lambda.80, spca.cv.90[1,which.min(spca.cv.90[3,])], num.vox.vec)
# 
# #Plotting 
# Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (spca.cv.90[1,which.min(spca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
# pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
# beta_pms <- pre_beta_pms%*%age_tab$age
# rownames(beta_pms) <- colnames(sub.dat)
# rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
# 
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0] <- abs(c(beta_pms))
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_SPCA_cov'))
# var.sel <- (order(abs(beta_pms), decreasing=TRUE))
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
# mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_pm_smooth_SPCA_cov_rank'))
# 
# remove(lambda.80)
# remove(pre_beta_pms)
# remove(mask.temp)
# remove(beta_pms)
# remove(Omeg)

#Need to save saliency and


out <- rbind(res.ridge$train,res.lambda.pca.90$train,res.lambda.pca.90.p$train, res.lambda.pca.90.c$train,
             res.ridge$test,res.lambda.pca.90$test,res.lambda.pca.90.p$test,res.lambda.pca.90.c$test)

write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/oct29_pm_pms90_smooth_shrink_pca.csv'), row.names = FALSE)





