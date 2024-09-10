#Aug 30, introduce a double cv. (one during screening, one during pred which shouldn't matter), but more importantly, correctly cv during screening for ridge

#Sep 2, instead of adding signals to raw data, i'll replace signals

#sep4 ==> assessing shrinkage on Lambda. Look at _4_ and _5_

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
################################################################ pca no-smooth ######################################################################
print("load data")
##Get age response
part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
#These two are equal
part_use<-part_list[part_list$exist_vbm==1,] #4262 participants left
part_use<-part_use[1:1000,] #only take 4k

agetab<-read.table(file = '/well/nichols/projects/UKB/SMS/ukb_latest-Age.tsv', sep = '\t', header = TRUE)
age_tab<-as.data.frame(matrix(,nrow = length(part_use$V1),ncol = 2)) #id, age, number of masked voxels
colnames(age_tab)[1:2]<-c('id','age')
age_tab$id<-part_use$V1
age_tab.test <- age_tab[501:1000,]

# age_tab <- age_tab[1:2000,]
age_tab <- age_tab[1:500,]

n.train <- nrow(age_tab)

##Get data 

list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
# sub.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
sub.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
print("here 1")

column.names <- as.character(1:ncol(sub.dat))
print("here 2")

colnames(sub.dat) <- as.character(1:ncol(sub.dat))
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab.test$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
# sub.dat.test <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
sub.dat.test <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
print("here 3")

colnames(sub.dat.test) <- colnames(sub.dat)

##################################################################
# trainset <- list(a = trainset.ind, b = trainset.ind2)
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')

mask.vec <- c(fast_read_imgs_mask('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz','/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))

#Defining signal to be around region 4 and 15 
mask.index <- (mask.vec %in% c(4,15))
mask.signal <- which(mask.index) #

set.seed(4)
sub.dat[,mask.signal] <- matrix(rnorm(nrow(sub.dat)*length(mask.signal),0,0.1) , nrow = nrow(sub.dat), ncol = length(mask.signal))
set.seed(4)
sub.dat.test[,mask.signal] <-  matrix(rnorm(nrow(sub.dat.test)*length(mask.signal),0,0.1) , nrow = nrow(sub.dat.test), ncol = length(mask.signal))


signal.strs <- c(0,0.5,3)
signal.strs.diff <- c(0,diff(signal.strs))

for(q in (1:length(signal.strs))){
  
  signal.str <- signal.strs[q]
  
  print(paste0("########################.   Signal  ",signal.str))
  
  #### Change signal here
  sub.dat[,mask.signal] <- sub.dat[,mask.signal]  + signal.strs.diff[q]
  
  sub.dat.test[,mask.signal] <-  sub.dat.test[,mask.signal] + signal.strs.diff[q] 
  
  signal.sum.func <- function(dat,indices){
    return(sum(dat[indices]))
  }
  
  res.var <- apply(sub.dat, MARGIN = 1, FUN = signal.sum.func, indices = mask.signal)
  res.var.test <- apply(sub.dat.test, MARGIN = 1, FUN = signal.sum.func, indices = mask.signal)
  
  ################################################################ pca smooth 3 ######################################################################
  smooth_1d_image <- function(x){
    #turn 1d image into 3d
    gp.mask.hs <- res3.mask
    gp.mask.hs[gp.mask.hs!=0] <- x
    #smoothen
    gp.mask.hs <- gaussianSmooth(gp.mask.hs, c(1,1,1))
    #mask it again
    gp.mask.hs <- gp.mask.hs[res3.mask!=0]
    return(gp.mask.hs)
  }
  sub.dat.s <- t(apply(sub.dat,MARGIN = 1, smooth_1d_image))
  sub.dat.s.test <- t(apply(sub.dat.test,MARGIN = 1, smooth_1d_image))
  
  colnames(sub.dat.s) <- column.names
  colnames(sub.dat.s.test) <- column.names
  
  colnames(sub.dat) <- column.names
  colnames(sub.dat.test) <- column.names
  
  
  print("ridge")
  #Ridge
  num.vox.vec <- (1:40)*100
  
  theta.range <- 10^seq(-1,3,1)
  
  simple.cv <- function(lambda, no.variable = 1000,theta.range. = theta.range){ #Changed from -9 to -8, to -7 for 2k
    train <- vector(mode = 'numeric')
    test <- vector(mode = 'numeric')
    for(theta in theta.range.){
      #doing PCA
      Omeg <- solve(sub.dat.s%*%lambda%*%t(sub.dat.s) + (theta)*diag(nrow(sub.dat))) #4x4 
      pre_beta_pms <- lambda%*%t(sub.dat.s)%*%Omeg #4 x 75
      beta_pms <- pre_beta_pms%*%res.var
      rownames(beta_pms) <- colnames(sub.dat.s)
      rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
      
      #Doing prediction
      var.sel<- names(rank_beta_pms[1:no.variable])
      fit.ridge <- cv.glmnet(sub.dat[,var.sel],res.var, alpha=0)
      beta <- coef(fit.ridge, s = "lambda.min")
      train <- c(train,sqrt(mean((as.numeric(res.var - cbind(1,sub.dat[,var.sel])%*%beta))^2)))
      test <- c(test,sqrt(mean((as.numeric(res.var.test - cbind(1,sub.dat.test[,var.sel])%*%beta))^2)))
    }
    out <- rbind(theta.range,train,test)
    return(out)
  }
  
  #below is practically long.rmse
  print("making predictions")
  long.rmse.with.lambda.theta.ridge <- function(lambda,theta,num.vox.vec){
    Omeg <- solve(sub.dat%*%lambda%*%t(sub.dat.s) + (theta)*diag(nrow(sub.dat))) #4x4 
    pre_beta_pms <- lambda%*%t(sub.dat.s)%*%Omeg #4 x 75
    beta_pms <- pre_beta_pms%*%res.var
    rownames(beta_pms) <- colnames(sub.dat.s)
    rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
    train <- vector(mode = 'numeric')
    test <- vector(mode = 'numeric')
    for(i in num.vox.vec){
      print(i)
      var.sel<- names(rank_beta_pms[1:i])
      #ridge prediction
      # fit.ridge <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0)
      fit.ridge <- cv.glmnet(sub.dat[,var.sel],res.var, alpha=0) #fixing lambda
      # beta <- coef(fit.ridge, s = "lambda.min")
      beta <- coef(fit.ridge, s = "lambda.min")
      train <- c(train,sqrt(mean((as.numeric(res.var - cbind(1,sub.dat[,var.sel])%*%beta))^2)))
      test <- c(test,sqrt(mean((as.numeric(res.var.test - cbind(1,sub.dat.test[,var.sel])%*%beta))^2)))
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
  
  print("tune pca 90")
  lambda.80 <- pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90])
  
  #########
  print("summary pca rotation")
  print(summary(c(lambda.80)))
  max.cov <- max(abs(c(lambda.80)))
  if(max.cov > 1){
    shrinkage.param <- 1/max.cov
  } else{
    shrinkage.param <- 0.1
  }
  
  lambda.80 <- shrinkage.param*lambda.80 + (1-shrinkage.param)*diag(ncol(lambda.80))
  #########
  #save col means
  
  pca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
  print("rmse pca 90")
  res.lambda.pca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, pca.cv.90[1,which.min(pca.cv.90[3,])], num.vox.vec)
  
  #Plotting 
  Omeg <- solve(sub.dat.s%*%lambda.80%*%t(sub.dat.s) + (pca.cv.90[1,which.min(pca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
  pre_beta_pms <- lambda.80%*%t(sub.dat.s)%*%Omeg #4 x 75
  beta_pms <- pre_beta_pms%*%res.var
  rownames(beta_pms) <- colnames(sub.dat)
  rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
  
  mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  mask.temp[mask.temp!=0] <- abs(c(beta_pms))
  mask.temp@datatype = 16
  mask.temp@bitpix = 32
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/sep3_500_binmask_4_str',signal.str,'_PCA90_smooth'))
  ###
  
  # var.sel<- as.numeric(c(rank_beta_pms))
  var.sel <- (order(abs(beta_pms), decreasing=TRUE))
  
  #
  print(length(c(var.sel)))
  #
  mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)
  mask.temp@datatype = 16
  mask.temp@bitpix = 32
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/sep3_500_binmask_4_rank_str',signal.str,'_PCA90_smooth'))
  
  
  print("tune SPCA")
  
  cv_model <- cv.glmnet(sub.dat.s, res.var, alpha = 0, nfolds = 10)
  # Best lambda from cross-validation
  coefficients <- coef(cv_model, s = "lambda.min")  # Extract coefficients at best lambda
  weights <- abs(coefficients[-1,])  # Exclude intercept from weights (first row of coefficients)
  X_weighted <- scale(sub.dat.s) * sqrt(weights)
  # Perform PCA on the weighted matrix
  pca<- prcomp(X_weighted, center = TRUE, scale. = FALSE)
  
  explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
  # Calculate the cumulative variance explained
  cumulative_variance <- cumsum(explained_variance)
  # Determine the number of components needed to explain at least 90% of the variance
  num_components.90 <- which(cumulative_variance >= 0.90)[1]
  
  
  print("tune pca 90")
  lambda.80 <- pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90])
  
  #########
  print("summary Spca rotation")
  print(summary(c(lambda.80)))
  max.cov <- max(abs(c(lambda.80)))
  if(max.cov > 1){
    shrinkage.param <- 1/max.cov
  } else{
    shrinkage.param <- 0.1
  }
  
  lambda.80 <- shrinkage.param*lambda.80 + (1-shrinkage.param)*diag(ncol(lambda.80))
  #########
  
  #save col means
  spca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
  print("rmse SPCA 90")
  res.lambda.spca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, spca.cv.90[1,which.min(spca.cv.90[3,])], num.vox.vec)
  
  #Plotting 
  Omeg <- solve(sub.dat.s%*%lambda.80%*%t(sub.dat.s) + (spca.cv.90[1,which.min(spca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
  pre_beta_pms <- lambda.80%*%t(sub.dat.s)%*%Omeg #4 x 75
  beta_pms <- pre_beta_pms%*%res.var
  rownames(beta_pms) <- colnames(sub.dat)
  rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
  
  
  mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  mask.temp[mask.temp!=0] <- abs(c(beta_pms))
  mask.temp@datatype = 16
  mask.temp@bitpix = 32
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/sep3_500_binmask_4_str',signal.str,'_SPCA90_smooth'))
  
  # var.sel<- as.numeric(c(rank_beta_pms))
  var.sel <- (order(abs(beta_pms), decreasing=TRUE))
  
  mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)
  mask.temp@datatype = 16
  mask.temp@bitpix = 32
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/sep3_500_binmask_4_rank_str',signal.str,'_SPCA90_smooth'))
  
  ################################################################   ################################################################ 
  print("Fix shrinkage ########")
  
  print("tune pca")
  pca <-  prcomp(sub.dat)
  explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
  # Calculate the cumulative variance explained
  cumulative_variance <- cumsum(explained_variance)
  # Determine the number of components needed to explain at least 90% of the variance
  num_components.90 <- which(cumulative_variance >= 0.90)[1]
  
  print("tune pca 90")
  lambda.80 <- pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90])
  
  #########
  shrinkage.param <- 0.05
  
  lambda.80 <- shrinkage.param*lambda.80 + (1-shrinkage.param)*diag(ncol(lambda.80))
  #########
  #save col means
  
  pca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
  print("rmse pca 90")
  res.lambda.pca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, pca.cv.90[1,which.min(pca.cv.90[3,])], num.vox.vec)
  
  #Plotting 
  Omeg <- solve(sub.dat.s%*%lambda.80%*%t(sub.dat.s) + (pca.cv.90[1,which.min(pca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
  pre_beta_pms <- lambda.80%*%t(sub.dat.s)%*%Omeg #4 x 75
  beta_pms <- pre_beta_pms%*%res.var
  rownames(beta_pms) <- colnames(sub.dat)
  rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
  
  mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  mask.temp[mask.temp!=0] <- abs(c(beta_pms))
  mask.temp@datatype = 16
  mask.temp@bitpix = 32
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/sep3_500_binmask_5_str',signal.str,'_PCA90_smooth'))
  ###
  
  # var.sel<- as.numeric(c(rank_beta_pms))
  var.sel <- (order(abs(beta_pms), decreasing=TRUE))
  
  #
  print(length(c(var.sel)))
  #
  mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)
  mask.temp@datatype = 16
  mask.temp@bitpix = 32
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/sep3_500_binmask_5_rank_str',signal.str,'_PCA90_smooth'))
  
  
  print("tune SPCA")
  
  cv_model <- cv.glmnet(sub.dat.s, res.var, alpha = 0, nfolds = 10)
  # Best lambda from cross-validation
  coefficients <- coef(cv_model, s = "lambda.min")  # Extract coefficients at best lambda
  weights <- abs(coefficients[-1,])  # Exclude intercept from weights (first row of coefficients)
  X_weighted <- scale(sub.dat.s) * sqrt(weights)
  # Perform PCA on the weighted matrix
  pca<- prcomp(X_weighted, center = TRUE, scale. = FALSE)
  
  explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
  # Calculate the cumulative variance explained
  cumulative_variance <- cumsum(explained_variance)
  # Determine the number of components needed to explain at least 90% of the variance
  num_components.90 <- which(cumulative_variance >= 0.90)[1]
  
  
  print("tune pca 90")
  lambda.80 <- pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90])
  
  #########
  shrinkage.param <- 0.05
  
  lambda.80 <- shrinkage.param*lambda.80 + (1-shrinkage.param)*diag(ncol(lambda.80))
  #########
  
  #save col means
  spca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
  print("rmse SPCA 90")
  res.lambda.spca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, spca.cv.90[1,which.min(spca.cv.90[3,])], num.vox.vec)
  
  #Plotting 
  Omeg <- solve(sub.dat.s%*%lambda.80%*%t(sub.dat.s) + (spca.cv.90[1,which.min(spca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
  pre_beta_pms <- lambda.80%*%t(sub.dat.s)%*%Omeg #4 x 75
  beta_pms <- pre_beta_pms%*%res.var
  rownames(beta_pms) <- colnames(sub.dat)
  rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
  
  
  mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  mask.temp[mask.temp!=0] <- abs(c(beta_pms))
  mask.temp@datatype = 16
  mask.temp@bitpix = 32
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/sep3_500_binmask_5_str',signal.str,'_SPCA90_smooth'))
  
  # var.sel<- as.numeric(c(rank_beta_pms))
  var.sel <- (order(abs(beta_pms), decreasing=TRUE))
  
  mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)
  mask.temp@datatype = 16
  mask.temp@bitpix = 32
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/sep3_500_binmask_5_rank_str',signal.str,'_SPCA90_smooth'))


  
}