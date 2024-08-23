
#19 Apr: use *Centre mask* and 1000 subjects.

#27 may, mix all 5 methods for variable screening here. use ridge for predictions


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

##########
##########
print("load data")
##Get age response
part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
#These two are equal
part_use<-part_list[part_list$exist_vbm==1,] #4262 participants left
# part_use<-part_use[1:200,] #only take 200
# part_use<-part_use[1:4000,] #only take 4k
part_use<-part_use[1:1000,] #only take 4k


agetab<-read.table(file = '/well/nichols/projects/UKB/SMS/ukb_latest-Age.tsv', sep = '\t', header = TRUE)
age_tab<-as.data.frame(matrix(,nrow = length(part_use$V1),ncol = 2)) #id, age, number of masked voxels
colnames(age_tab)[1:2]<-c('id','age')
age_tab$id<-part_use$V1
# for(i in 1:length(part_use$V1)){
#   age_tab$age[i]<-agetab$X21003.2.0[agetab$eid_8107==sub(".", "",age_tab$id[i])]
# }
# age_tab.test <- age_tab[101:200,]
# age_tab <- age_tab[1:100,]
# age_tab.test <- age_tab[2001:4000,]
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

print("here 4")
####Calling mmand
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')

mask.vec <- c(fast_read_imgs_mask('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz','/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))

#Defining signal to be around region 4 and 15 
mask.signal <- which(mask.vec %in% c(4,15))

signal.str <- 3

sub.dat[,mask.signal] <- sub.dat[,mask.signal] + signal.str

sub.dat.test[,mask.signal] <- sub.dat.test[,mask.signal] + signal.str

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- colMeans(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug14_500_str',signal.str,'_meantrainingdata'))

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- colMeans(sub.dat.test)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug14_500_str',signal.str,'_meantestdata'))


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
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug14_500_str',signal.str,'_ridge'))

var.sel <- (order(abs(beta_no_int), decreasing=TRUE))[1:1867]
print(length(c(var.sel)))
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- 1
# mask.temp[mask.temp!=0][c(var.sel)] <- 2
# Create an index of non-zero elements
non_zero_indices <- which(mask.temp != 0, arr.ind = TRUE)
# Subset the non-zero indices using var.sel
selected_indices <- non_zero_indices[var.sel, , drop = FALSE]
# Assign the value of 2 to the selected indices in the original mask.temp
mask.temp[selected_indices] <- 2

mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug14_500_top_str',signal.str,'_ridge'))

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

# print("tune robust pca")
# 
# pca <- PcaHubert(sub.dat,kmax=n.train) 
# lambda. <- pca$loadings %*% t(pca$loadings) 
# 
# robpca.cv <- simple.cv(lambda =lambda.,no.variable = 1000,theta.range. = theta.range )


# 
# print("tune PLS")
# pls_model <- plsr(age_tab$age ~ sub.dat)
# lambda. <- loadings(pls_model) %*% t(loadings(pls_model)) #15847 x 15847
# pls.cv <- simple.cv(lambda.) #theta = 10 is optimal
# 





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
# num_components.80 <- which(cumulative_variance >= 0.80)[1]
# 
# 
# print("tune pca 80")
# lambda.80 <- pca$rotation[, 1:num_components.80]%*% t(pca$rotation[, 1:num_components.80])
# pca.cv.80 <- simple.cv(lambda.80) #theta = 10 is optimal
# print("rmse pca 80")
# res.lambda.pca.80 <- long.rmse.with.lambda.theta.ridge(lambda.80, pca.cv.80[1,which.min(pca.cv.80[3,])], num.vox.vec)
# 
# 
# #Plotting 
# Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (pca.cv.80[1,which.min(pca.cv.80[3,])])*diag(nrow(sub.dat))) #4x4
# pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
# beta_pms <- pre_beta_pms%*%age_tab$age
# rownames(beta_pms) <- colnames(sub.dat)
# rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
# 
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
# mask.temp[mask.temp!=0] <- abs(c(beta_pms))
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/july20_smooth_PCA80'))


####here modification 2 aug, add mu to lamda

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
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug14_500_str',signal.str,'_PCA90'))
###

var.sel <- (order(abs(beta_pms), decreasing=TRUE))[1:1867]

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- 1
# mask.temp[mask.temp!=0][c(var.sel)] <-2
non_zero_indices <- which(mask.temp != 0, arr.ind = TRUE)
# Subset the non-zero indices using var.sel
selected_indices <- non_zero_indices[var.sel, , drop = FALSE]
# Assign the value of 2 to the selected indices in the original mask.temp
mask.temp[selected_indices] <- 2
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug14_500_top_str',signal.str,'_PCA90'))


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
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug14_500_str',signal.str,'_SPCA90'))

###
var.sel <- (order(abs(beta_pms), decreasing=TRUE))[1:1867]

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- 1
# mask.temp[mask.temp!=0][c(var.sel)] <-2
non_zero_indices <- which(mask.temp != 0, arr.ind = TRUE)
# Subset the non-zero indices using var.sel
selected_indices <- non_zero_indices[var.sel, , drop = FALSE]
# Assign the value of 2 to the selected indices in the original mask.temp
mask.temp[selected_indices] <- 2
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug14_500_top_str',signal.str,'_SPCA90'))

#Need to save saliency and


out <- rbind(res.lambda.pca.90$train,res.lambda.spca.90$train,res.ridge$train, 
             res.lambda.pca.90$test,res.lambda.spca.90$test,res.ridge$test)
# print(paste0("ROBUST optimal Theta: ", robpca.cv[1,which.min(robpca.cv[3,])]))
# print(paste0("PCA optimal Theta: ", pca.cv[1,which.min(pca.cv[3,])]))
# print(paste0("pls optimal Theta: ",pls.cv[1,which.min(pls.cv[3,])]))
# print(paste0("SPCA optimal Theta: ", spca.cv[1,which.min(spca.cv[3,])]))
write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/sim_aug14_500_str',signal.str,'_pms90.csv'), row.names = FALSE)

