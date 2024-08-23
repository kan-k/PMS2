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
signal.strs <- c(2,5)

for(signal.str in signal.strs){

print("here 4")
####Calling mmand
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')

mask.vec <- c(fast_read_imgs_mask('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz','/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))

#Defining signal to be around region 4 and 15 
mask.index <- (mask.vec %in% c(4,15))
mask.signal <- which(mask.index) #


sub.dat[,mask.signal] <- sub.dat[,mask.signal] + signal.str

sub.dat.test[,mask.signal] <- sub.dat.test[,mask.signal] + signal.str

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- colMeans(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_str',signal.str,'_meantrainingdata'))

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- colMeans(sub.dat.test)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_str',signal.str,'_meantestdata'))


signal.sum.func <- function(dat,indices){
  return(sum(dat[indices]))
}

res.var <- apply(sub.dat, MARGIN = 1, FUN = signal.sum.func, indices = mask.signal)
res.var.test <- apply(sub.dat.test, MARGIN = 1, FUN = signal.sum.func, indices = mask.signal)


print("ridge")
#Ridge

set.seed(4)
fit.ridge <- cv.glmnet(sub.dat,res.var, alpha=0)
beta <- coef(fit.ridge)
beta_no_int <- beta[-1,]
rank_beta.ridge <- beta_no_int[order(abs(beta_no_int), decreasing=TRUE)]
num.vox.vec <- (1:40)*100

long.rmse <- function(ranked_coef, num.vox.vec){
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(i in num.vox.vec){
    var.sel<- names(ranked_coef[1:i])
    # fit.ridge.pca <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0)
    fit.ridge.pca <- glmnet(sub.dat[,var.sel],res.var, alpha=0, lambda = 1e-2) #fixing lambda
    beta <- coef(fit.ridge.pca)
    train <- c(train,sqrt(mean((as.numeric(res.var - cbind(1,sub.dat[,var.sel])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(res.var.test- cbind(1,sub.dat.test[,var.sel])%*%beta))^2)))
  }
  out <- list()
  out$train <- train
  out$test <- test
  return(out)
}

res.ridge <- long.rmse(rank_beta.ridge, num.vox.vec)

# Full Ridge
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_no_int))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_str',signal.str,'_ridge'))

var.sel <- (order(abs(beta_no_int), decreasing=TRUE))
print(length(c(var.sel)))
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_rank_str',signal.str,'_ridge'))

# theta.range <- 10^seq(-4,3,1)

theta.range <- 10^seq(-1,3,1)


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
    beta_pms <- pre_beta_pms%*%res.var
    print(4)
    rownames(beta_pms) <- colnames(sub.dat)
    rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
    
    #Doing prediction
    var.sel<- names(rank_beta_pms[1:no.variable])
    print(5)
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
  Omeg <- solve(sub.dat%*%lambda%*%t(sub.dat) + (theta)*diag(nrow(sub.dat))) #4x4 
  pre_beta_pms <- lambda%*%t(sub.dat)%*%Omeg #4 x 75
  beta_pms <- pre_beta_pms%*%res.var
  rownames(beta_pms) <- colnames(sub.dat)
  rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(i in num.vox.vec){
    print(i)
    var.sel<- names(rank_beta_pms[1:i])
    #ridge prediction
    # fit.ridge <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0)
    fit.ridge <- glmnet(sub.dat[,var.sel],res.var, alpha=0, lambda = 1e-2) #fixing lambda
    # beta <- coef(fit.ridge, s = "lambda.min")
    beta <- coef(fit.ridge)
    train <- c(train,sqrt(mean((as.numeric(res.var - cbind(1,sub.dat[,var.sel])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(res.var.test - cbind(1,sub.dat.test[,var.sel])%*%beta))^2)))
  }
  out <- list()
  out$train <- train
  out$test <- test
  return(out)
}


# num.vox.vec <- (1:50)*200

# pca <- PcaHubert(sub.dat,kmax=100) 
print("rmse robpca")

# pca <- PcaHubert(sub.dat,kmax=n.train) 
# lambda. <- pca$loadings %*% t(pca$loadings) 
# print(paste0("dim ROBPCA: ", dim(pca$loadings)))
# res.lambda.ropca <- long.rmse.with.lambda.theta.ridge(lambda., robpca.cv[1,which.min(robpca.cv[3,])], num.vox.vec)
print("tune pca")
pca <-  prcomp(sub.dat)
explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
# Calculate the cumulative variance explained
cumulative_variance <- cumsum(explained_variance)
# Determine the number of components needed to explain at least 90% of the variance
num_components.90 <- which(cumulative_variance >= 0.90)[1]

print("tune pca 90")
lambda.80 <- pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90])
#save col means

pca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
print("rmse pca 90")
res.lambda.pca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, pca.cv.90[1,which.min(pca.cv.90[3,])], num.vox.vec)

#Plotting 
Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (pca.cv.90[1,which.min(pca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
beta_pms <- pre_beta_pms%*%res.var
rownames(beta_pms) <- colnames(sub.dat)
rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_pms))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_str',signal.str,'_PCA90'))
###

var.sel <- (order(abs(beta_pms), decreasing=TRUE))

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_rank_str',signal.str,'_PCA90'))


print("tune SPCA")

cv_model <- cv.glmnet(sub.dat, res.var, alpha = 0, nfolds = 10)
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
#save col means
spca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
print("rmse SPCA 90")
res.lambda.spca.90 <- long.rmse.with.lambda.theta.ridge(lambda.80, spca.cv.90[1,which.min(spca.cv.90[3,])], num.vox.vec)

#Plotting 
Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (spca.cv.90[1,which.min(spca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
beta_pms <- pre_beta_pms%*%res.var
rownames(beta_pms) <- colnames(sub.dat)
rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_pms))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_str',signal.str,'_SPCA90'))

###
var.sel <- (order(abs(beta_pms), decreasing=TRUE))

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_rank_str',signal.str,'_SPCA90'))

#Need to save saliency and


out <- rbind(res.lambda.pca.90$train,res.lambda.spca.90$train,res.ridge$train, 
             res.lambda.pca.90$test,res.lambda.spca.90$test,res.ridge$test)
write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/sim_aug17_500_binmask_str',signal.str,'_pms90.csv'), row.names = FALSE)



################################################################ PMS ######################################################################


res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')


#PMS
img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
nb <- find_brain_image_neighbors(img1, res3.mask, radius=1)

n.train <- nrow(sub.dat)
n.test <- nrow(sub.dat.test)
rho_high <- rep(-log(0.9)/4,length=ncol(sub.dat))

num.vox.vec <- (1:40)*100



print("stage 3")

#pms high
out.pms.hi<-fast_PMS_local_spatial(x=as.matrix(sub.dat), y = res.var, coords=nb$maskcoords,neighbors=nb$mask_img_nb,num_neighbors=nb$num_neighbors, rho = rho_high)$pms_select
#holp
out.holp<-fast_PMS_cpp(x=as.matrix(sub.dat), y = res.var, theta = 0)$pms_select

#pms bootstrap
for(i in num.vox.vec){
  temp_pms<-vector(mode = 'integer')
  ####below is the loop for repeated pms for *robust/bootstrapping* estimate
  for(m in 1:10){ #10 times
    sam.ind <- sample(1:n.train,size = 100)
    temp_pms <- c(temp_pms, fast_PMS_local_spatial(x=as.matrix(sub.dat[sam.ind,]), y = res.var[sam.ind], coords=nb$maskcoords,neighbors=nb$mask_img_nb,num_neighbors=nb$num_neighbors, rho = rho_high)$pms_select[1:i])
  }
  pms_table <- table(temp_pms)
  out.pms.boot <- as.integer(names(sort(pms_table,decreasing = TRUE)))
}


long.rmse.pms <- function(pms_ind,num.vox.vec){ #this function is extremely similar to long.rmse for ridge, but without "names( ind )"
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(i in num.vox.vec){
    print(i)
    var.sel<- pms_ind[1:i]
    #ridge prediction
    # fit.ridge <- cv.glmnet(sub.dat[,var.sel],res.var, alpha=0)
    fit.ridge <- glmnet(sub.dat[,var.sel],res.var, alpha=0, lambda = 1e-2) #fixing lambda
    # beta <- coef(fit.ridge, s = "lambda.min")
    beta <- coef(fit.ridge)
    train <- c(train,sqrt(mean((as.numeric(res.var - cbind(1,sub.dat[,var.sel])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(res.var.test - cbind(1,sub.dat.test[,var.sel])%*%beta))^2)))
  }
  out <- list()
  out$train <- train
  out$test <- test
  return(out)
}

res.pms.hi <- long.rmse.pms(out.pms.hi, num.vox.vec)
res.pms.holp <- long.rmse.pms(out.holp, num.vox.vec)
res.pms.boot <- long.rmse.pms(out.pms.boot, num.vox.vec)

out <- rbind(res.pms.hi$train,res.pms.holp$train ,res.pms.boot$train, 
             res.pms.hi$test,res.pms.holp$test,res.pms.boot$test)

write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/sim_aug17_500_binmask_str',signal.str,'_nbpms.csv'), row.names = FALSE)

#plot out.pms.hi
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0][out.pms.hi] <- c(ncol(sub.dat):1)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_rank_str',signal.str,'_pmshi'))

#plot out.pms.holp
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0][out.holp] <- c(ncol(sub.dat):1)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_rank_str',signal.str,'_holp'))



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

set.seed(4)
fit.ridge <- cv.glmnet(sub.dat.s,res.var, alpha=0)
beta <- coef(fit.ridge)
beta_no_int <- beta[-1,]
rank_beta.ridge <- beta_no_int[order(abs(beta_no_int), decreasing=TRUE)]
num.vox.vec <- (1:40)*100

long.rmse <- function(ranked_coef, num.vox.vec){
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(i in num.vox.vec){
    var.sel<- names(ranked_coef[1:i])
    # fit.ridge.pca <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0)
    fit.ridge.pca <- glmnet(sub.dat[,var.sel],res.var, alpha=0, lambda = 1e-2) #fixing lambda
    beta <- coef(fit.ridge.pca)
    train <- c(train,sqrt(mean((as.numeric(res.var - cbind(1,sub.dat[,var.sel])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(res.var.test- cbind(1,sub.dat.test[,var.sel])%*%beta))^2)))
  }
  out <- list()
  out$train <- train
  out$test <- test
  return(out)
}

res.ridge <- long.rmse(rank_beta.ridge, num.vox.vec)

# Full Ridge
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_no_int))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_3_str',signal.str,'_ridge_smooth'))

# var.sel<- as.numeric(rank_beta.ridge)
var.sel <- (order(abs(beta_no_int), decreasing=TRUE))
print(length(c(var.sel)))
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_3_rank_str',signal.str,'_ridge_smooth'))

theta.range <- 10^seq(-1,3,1)

simple.cv <- function(lambda, no.variable = 1000,theta.range. = theta.range){ #Changed from -9 to -8, to -7 for 2k
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(theta in theta.range.){
    #doing PCA
    print(1)
    Omeg <- solve(sub.dat.s%*%lambda%*%t(sub.dat.s) + (theta)*diag(nrow(sub.dat))) #4x4 
    print(2)
    pre_beta_pms <- lambda%*%t(sub.dat.s)%*%Omeg #4 x 75
    print(3)
    beta_pms <- pre_beta_pms%*%res.var
    print(4)
    rownames(beta_pms) <- colnames(sub.dat.s)
    rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
    
    #Doing prediction
    var.sel<- names(rank_beta_pms[1:no.variable])
    print(5)
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
    fit.ridge <- glmnet(sub.dat[,var.sel],res.var, alpha=0, lambda = 1e-2) #fixing lambda
    # beta <- coef(fit.ridge, s = "lambda.min")
    beta <- coef(fit.ridge)
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
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_3_str',signal.str,'_PCA90_smooth'))
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
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_3_rank_str',signal.str,'_PCA90_smooth'))


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
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_3_str',signal.str,'_SPCA90_smooth'))

# var.sel<- as.numeric(c(rank_beta_pms))
var.sel <- (order(abs(beta_pms), decreasing=TRUE))

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_3_rank_str',signal.str,'_SPCA90_smooth'))

###
out <- rbind(res.lambda.pca.90$train,res.lambda.spca.90$train,res.ridge$train, 
             res.lambda.pca.90$test,res.lambda.spca.90$test,res.ridge$test)
write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/sim_aug17_500_binmask_3_str',signal.str,'_pms90_smooth.csv'), row.names = FALSE)


################################################################ pca smooth 2 ######################################################################

print("ridge")
#Ridge

set.seed(4)
fit.ridge <- cv.glmnet(sub.dat.s,res.var, alpha=0)
beta <- coef(fit.ridge)
beta_no_int <- beta[-1,]
rank_beta.ridge <- beta_no_int[order(abs(beta_no_int), decreasing=TRUE)]
num.vox.vec <- (1:40)*100

long.rmse <- function(ranked_coef, num.vox.vec){
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(i in num.vox.vec){
    var.sel<- names(ranked_coef[1:i])
    # fit.ridge.pca <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0)
    fit.ridge.pca <- glmnet(sub.dat.s[,var.sel],res.var, alpha=0, lambda = 1e-2) #fixing lambda
    beta <- coef(fit.ridge.pca)
    train <- c(train,sqrt(mean((as.numeric(res.var - cbind(1,sub.dat.s[,var.sel])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(res.var.test- cbind(1,sub.dat.s.test[,var.sel])%*%beta))^2)))
  }
  out <- list()
  out$train <- train
  out$test <- test
  return(out)
}

res.ridge <- long.rmse(rank_beta.ridge, num.vox.vec)

# Full Ridge
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_no_int))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_2_str',signal.str,'_ridge_smooth'))

# var.sel<- as.numeric(rank_beta.ridge)
var.sel <- (order(abs(beta_no_int), decreasing=TRUE))
print(length(c(var.sel)))
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_2_rank_str',signal.str,'_ridge_smooth'))

theta.range <- 10^seq(-1,3,1)

simple.cv <- function(lambda, no.variable = 1000,theta.range. = theta.range){ #Changed from -9 to -8, to -7 for 2k
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(theta in theta.range.){
    #doing PCA
    print(1)
    Omeg <- solve(sub.dat.s%*%lambda%*%t(sub.dat.s) + (theta)*diag(nrow(sub.dat))) #4x4 
    print(2)
    pre_beta_pms <- lambda%*%t(sub.dat.s)%*%Omeg #4 x 75
    print(3)
    beta_pms <- pre_beta_pms%*%res.var
    print(4)
    rownames(beta_pms) <- colnames(sub.dat.s)
    rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
    
    #Doing prediction
    var.sel<- names(rank_beta_pms[1:no.variable])
    print(5)
    fit.ridge <- cv.glmnet(sub.dat.s[,var.sel],res.var, alpha=0)
    beta <- coef(fit.ridge, s = "lambda.min")
    train <- c(train,sqrt(mean((as.numeric(res.var - cbind(1,sub.dat.s[,var.sel])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(res.var.test - cbind(1,sub.dat.s.test[,var.sel])%*%beta))^2)))
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
    fit.ridge <- glmnet(sub.dat.s[,var.sel],res.var, alpha=0, lambda = 1e-2) #fixing lambda
    # beta <- coef(fit.ridge, s = "lambda.min")
    beta <- coef(fit.ridge)
    train <- c(train,sqrt(mean((as.numeric(res.var - cbind(1,sub.dat.s[,var.sel])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(res.var.test - cbind(1,sub.dat.s.test[,var.sel])%*%beta))^2)))
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
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_2_str',signal.str,'_PCA90_smooth'))
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
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_2_rank_str',signal.str,'_PCA90_smooth'))


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
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_2_str',signal.str,'_SPCA90_smooth'))

# var.sel<- as.numeric(c(rank_beta_pms))
var.sel <- (order(abs(beta_pms), decreasing=TRUE))

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug17_500_binmask_2_rank_str',signal.str,'_SPCA90_smooth'))

###
out <- rbind(res.lambda.pca.90$train,res.lambda.spca.90$train,res.ridge$train, 
             res.lambda.pca.90$test,res.lambda.spca.90$test,res.ridge$test)
write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/sim_aug17_500_binmask_2_str',signal.str,'_pms90_smooth.csv'), row.names = FALSE)
}