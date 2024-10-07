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
install.packages("/well/nichols/users/qcv214/pms2/package/mmand_1.6.2.tar.gz", repos = NULL, type = "source")
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
# quantile_thresholds <- quantile(depind[train.test.ind$train], probs = seq(0, 1, by = 0.34))
# #age.group <- ifelse(age > mean(age), yes = 1, no = -1)
# dep.group1 <- ifelse(depind <=quantile_thresholds[[2]], yes =1, no = 0)
# dep.group2 <- ifelse(depind > quantile_thresholds[[2]] & depind <=quantile_thresholds[[3]], yes =1, no = 0)
# dep.group3 <- ifelse(depind > quantile_thresholds[[3]], yes =1, no = 0)
# 
# quantile_thresholds <- quantile(age[train.test.ind$train], probs = seq(0, 1, by = 0.34))
# #age.group <- ifelse(age > mean(age), yes = 1, no = -1)
# age.group1 <- ifelse(age <=quantile_thresholds[[2]], yes =1, no = 0)
# age.group2 <- ifelse(age > quantile_thresholds[[2]] & age <=quantile_thresholds[[3]], yes =1, no = 0)
# age.group3 <- ifelse(age > quantile_thresholds[[3]], yes =1, no = 0)
# 
# co.dat <- cbind(sex,dep.group1,dep.group2,dep.group3 ,age.group1,age.group2,age.group3)
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


######################################################################################################
#### Smoothing 2: smoothing only applied for screening, but NOT for prediction.


##############################################################################################################################
######Non-smooth
num.vox.vec <- (1:50)*200

print("ridge")
#Ridge

set.seed(4)
fit.ridge <- cv.glmnet(sub.dat,age_tab$age, alpha=0)
beta <- coef(fit.ridge, s = "lambda.min")
beta_no_int <- beta[-1,]
rank_beta.ridge <- beta_no_int[order(abs(beta_no_int), decreasing=TRUE)]

long.rmse <- function(ranked_coef, num.vox.vec){
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(i in num.vox.vec){
    var.sel<- names(ranked_coef[1:i])
    # fit.ridge.pca <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0)
    fit.ridge.pca <- cv.glmnet(cbind(sub.dat[,var.sel],co.dat[train.test.ind$train, ]),age_tab$age, alpha=0)  #fixing lambda
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
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sep24_pm_dcv_ridge'))
  
  var.sel <- (order(abs(beta_no_int), decreasing=TRUE))
  print(length(c(var.sel)))
  mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
  mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
  mask.temp@datatype = 16
  mask.temp@bitpix = 32
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sep24_pm_dcv_ridge_rank'))
  

theta.range <- 10^seq(-4,3,1)


#Robust PCA


simple.cv <- function(lambda, no.variable = 1000,theta.range. = theta.range){ #Changed from -9 to -8, to -7 for 2k
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(theta in theta.range.){
    #doing PCA
    print(1)
    Omeg <- solve(sub.dat%*%lambda%*%t(sub.dat) + (theta)*diag(nrow(sub.dat))) #4x4 
    print(2)
    pre_beta_pms <- lambda%*%t(sub.dat)%*%Omeg #4 x 75
    print(3)
    beta_pms <- pre_beta_pms%*%age_tab$age
    print(4)
    rownames(beta_pms) <- colnames(sub.dat)
    rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
    
    #Doing prediction
    var.sel<- names(rank_beta_pms[1:no.variable])
    print(5)
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
    print(i)
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


# pca <- PcaHubert(sub.dat,kmax=100) 
print("rmse robpca")
print("tune pca")
pca <-  prcomp(sub.dat)
explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
# Calculate the cumulative variance explained
cumulative_variance <- cumsum(explained_variance)
# Determine the number of components needed to explain at least 90% of the variance
num_components.90 <- which(cumulative_variance >= 0.90)[1]


print("tune pca 90")
lambda.80 <- pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90])
pca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
print("rmse pca 90")
res.lambda.pca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, pca.cv.90[1,which.min(pca.cv.90[3,])], num.vox.vec)

    Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (pca.cv.90[1,which.min(pca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
    pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
    beta_pms <- pre_beta_pms%*%age_tab$age
    rownames(beta_pms) <- colnames(sub.dat)
    rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
    
    mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
    mask.temp[mask.temp!=0] <- abs(c(beta_pms))
    mask.temp@datatype = 16
    mask.temp@bitpix = 32
    writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sep24_pm_dcv_PCA'))
    
    ###
    var.sel <- (order(abs(beta_pms), decreasing=TRUE))
    mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
    mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
    mask.temp@datatype = 16
    mask.temp@bitpix = 32
    writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sep24_pm_dcv_PCA_rank'))

    ####Removing previous data
    remove(lambda.80)
    remove(pre_beta_pms)
    remove(mask.temp)
    remove(beta_pms)
    remove(Omeg)


print("tune SPCA")

cv_model <- cv.glmnet(sub.dat, age_tab$age, alpha = 0, nfolds = 10)
# Best lambda from cross-validation
coefficients <- coef(cv_model, s = "lambda.min")  # Extract coefficients at best lambda
weights <- abs(coefficients[-1,])  # Exclude intercept from weights (first row of coefficients)
X_weighted <- scale(sub.dat) * sqrt(weights)
# Perform PCA on the weighted matrix
pca<- prcomp(X_weighted, center = TRUE, scale. = FALSE)

explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
# Calculate the cumulative variance explained
cumulative_variance <- cumsum(explained_variance)
# Determine the number of components needed to explain at least 90% of the variance
num_components.90 <- which(cumulative_variance >= 0.90)[1]

print("tune pca 90")
lambda.80 <- pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90])
spca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
print("rmse SPCA 90")
res.lambda.spca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, spca.cv.90[1,which.min(spca.cv.90[3,])], num.vox.vec)

Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (pca.cv.90[1,which.min(pca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
beta_pms <- pre_beta_pms%*%age_tab$age
rownames(beta_pms) <- colnames(sub.dat)
rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]

    mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
    mask.temp[mask.temp!=0] <- abs(c(beta_pms))
    mask.temp@datatype = 16
    mask.temp@bitpix = 32
    writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sep24_pm_dcv_SPCA'))
    
    ###
    var.sel <- (order(abs(beta_pms), decreasing=TRUE))
    mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
    mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)/ncol(sub.dat)
    mask.temp@datatype = 16
    mask.temp@bitpix = 32
    writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sep24_pm_dcv_SPCA_rank'))
    
    ####Removing previous data
    remove(lambda.80)
    remove(pre_beta_pms)
    remove(mask.temp)
    remove(beta_pms)
    remove(Omeg)
    remove(pca)
    remove(lambda.80)
    remove(weights)
    remove(X_weighted)


out <- rbind(res.lambda.pca.90$train,res.lambda.spca.90$train ,res.ridge$train, 
             res.lambda.pca.90$test,res.lambda.spca.90$test,res.ridge$test)
write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/sep24_pm_dcv_pms8090_full.csv'), row.names = FALSE)


##################################################################################################################################
#################### PMS and HOLP

#PMS
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id[1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
nb <- find_brain_image_neighbors(img1, res3.mask, radius=1)

n.train <- nrow(sub.dat)
n.test <- nrow(sub.dat.test)
rho_high <- rep(-log(0.9)/4,length=ncol(sub.dat))

print("stage 3")

#pms high
out.pms.hi<-fast_PMS_local_spatial(x=as.matrix(sub.dat), y = age_tab$age, coords=nb$maskcoords,neighbors=nb$mask_img_nb,num_neighbors=nb$num_neighbors, rho = rho_high)$pms_select
#holp
out.holp<-fast_PMS_cpp(x=as.matrix(sub.dat), y = age_tab$age, theta = 0)$pms_select


long.rmse.pms <- function(pms_ind,num.vox.vec){ #this function is extremely similar to long.rmse for ridge, but without "names( ind )"
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(i in num.vox.vec){
    print(i)
    var.sel<- pms_ind[1:i]
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

res.pms.hi <- long.rmse.pms(out.pms.hi, num.vox.vec)
res.pms.holp <- long.rmse.pms(out.holp, num.vox.vec)

out <- rbind(res.pms.hi$train,res.pms.holp$train, 
             res.pms.hi$test,res.pms.holp$test)

write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/sep24_pm_dcv_nbpms_full.csv'), row.names = FALSE)

#plot out.pms.hi
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0][out.pms.hi] <- c(ncol(sub.dat):1)/ncol(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sep24_pm_dcv_pms_hi'))

#plot out.pms.holp
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz')
mask.temp[mask.temp!=0][out.holp] <- c(ncol(sub.dat):1)/ncol(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sep24_pm_dcv_pms_holp'))


