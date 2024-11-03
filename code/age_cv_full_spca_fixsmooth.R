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

theta.range <- c(1e-4,1e-3,1e-1,1,1e3)


##################################################################################################################################
#################### PMS and HOLP

########################################################################################################################################################
################################################################ Smoothing #############################################################################

smoothing.sigmas <-  2#c(0.5,2)

for(smoothing.sigma in smoothing.sigmas){
  
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
    fit.ridge <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0)
    beta <- coef(fit.ridge, s = "lambda.min")
    train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat[,var.sel])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test[,var.sel])%*%beta))^2)))
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
    fit.ridge <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0) #fixing lambda
    # beta <- coef(fit.ridge, s = "lambda.min")
    beta <- coef(fit.ridge, s = "lambda.min")
    train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat[,var.sel])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test[,var.sel])%*%beta))^2)))
  }
  out <- list()
  out$train <- train
  out$test <- test
  return(out)
}

##################################################
print("tune pca 90")

shrinkage.param <- 0.3
# 

################################################################################ 

print("tune SPCA")

cv_model <- glmnet(sub.dat, age_tab$age, alpha = 0, lambda=1e-4)
# Best lambda from cross-validation
coefficients <- coef(cv_model)  # Extract coefficients at best lambda
weights <- abs(coefficients[-1,])  # Exclude intercept from weights (first row of coefficients)
X_weighted <- scale(sub.dat) * sqrt(weights)
# Perform PCA on the weighted matrix
pca<- prcomp(X_weighted, center = TRUE, scale. = FALSE)

explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
# Calculate the cumulative variance explained
cumulative_variance <- cumsum(explained_variance)
# Determine the number of components needed to explain at least 90% of the variance
num_components.90 <- which(cumulative_variance >= 0.90)[1]


################################################## No shrinkage
lambda.80 <- (pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90]))

spca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
print("rmse SPCA 90")
res.lambda.spca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, spca.cv.90[1,which.min(spca.cv.90[3,])], num.vox.vec)

#Plotting 
Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (spca.cv.90[1,which.min(spca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
beta_pms <- pre_beta_pms%*%age_tab$age
rownames(beta_pms) <- colnames(sub.dat)
rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_pms))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_age_smooth',smoothing.sigma,'_SPCA'))
var.sel <- (order(abs(beta_pms), decreasing=TRUE))
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_age_smooth',smoothing.sigma,'_SPCA_rank'))

####Removing previous data
remove(lambda.80)
remove(pre_beta_pms)
remove(mask.temp)
remove(beta_pms)
remove(Omeg)
################################################## 



print("tune pca 90")
lambda.80 <- shrinkage.param*(pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90]))
diag(lambda.80) <- diag(lambda.80) + (1-shrinkage.param)

spca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
print("rmse SPCA 90")
res.lambda.spca.90.p <- long.rmse.with.lambda.theta.ridge(lambda.80, spca.cv.90[1,which.min(spca.cv.90[3,])], num.vox.vec)

#Plotting 
Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (spca.cv.90[1,which.min(spca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
beta_pms <- pre_beta_pms%*%age_tab$age
rownames(beta_pms) <- colnames(sub.dat)
rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_pms))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_age_smooth',smoothing.sigma,'_SPCA_proj'))
var.sel <- (order(abs(beta_pms), decreasing=TRUE))
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_age_smooth',smoothing.sigma,'_SPCA_proj_rank'))

####Removing previous data
remove(lambda.80)
remove(pre_beta_pms)
remove(mask.temp)
remove(beta_pms)
remove(Omeg)
###### Cov
###### Cov
lambda.80 <- shrinkage.param*pca$rotation[, 1:num_components.90] %*% diag(pca$sdev[1:num_components.90]^2) %*% t(pca$rotation[, 1:num_components.90])
diag(lambda.80) <- diag(lambda.80) + (1-shrinkage.param)

spca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
print("rmse SPCA 90")
res.lambda.spca.90.c <- long.rmse.with.lambda.theta.ridge(lambda.80, spca.cv.90[1,which.min(spca.cv.90[3,])], num.vox.vec)

#Plotting 
Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (spca.cv.90[1,which.min(spca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
beta_pms <- pre_beta_pms%*%age_tab$age
rownames(beta_pms) <- colnames(sub.dat)
rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_pms))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_age_smooth',smoothing.sigma,'_SPCA_cov'))
var.sel <- (order(abs(beta_pms), decreasing=TRUE))
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct29_age_smooth',smoothing.sigma,'_SPCA_cov_rank'))

remove(lambda.80)
remove(pre_beta_pms)
remove(mask.temp)
remove(beta_pms)
remove(Omeg)

#Need to save saliency and


out <- rbind(res.lambda.spca.90$train,res.lambda.spca.90.p$train ,res.lambda.spca.90.c$train,
             res.lambda.spca.90$test,res.lambda.spca.90.p$test,res.lambda.spca.90.c$test)

write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/oct29_age_pms90_smooth',smoothing.sigma,'_shrink_spca.csv'), row.names = FALSE)

}



