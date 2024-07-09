#Use res3mask with 150k voxels

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
p_load(robustbase)
p_load(stats)




##########
##########
print("load data")
##Get age response
part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
#These two are equal
part_use<-part_list[part_list$exist_vbm==1,] #4262 participants left
# part_use<-part_use[1:200,] #only take 200
part_use<-part_use[1:4000,] #only take 4k

agetab<-read.table(file = '/well/nichols/projects/UKB/SMS/ukb_latest-Age.tsv', sep = '\t', header = TRUE)
age_tab<-as.data.frame(matrix(,nrow = length(part_use$V1),ncol = 2)) #id, age, number of masked voxels
colnames(age_tab)[1:2]<-c('id','age')
age_tab$id<-part_use$V1
for(i in 1:length(part_use$V1)){
  age_tab$age[i]<-agetab$X21003.2.0[agetab$eid_8107==sub(".", "",age_tab$id[i])]
}
# age_tab.test <- age_tab[101:200,]
# age_tab <- age_tab[1:100,]
age_tab.test <- age_tab[2001:4000,]
age_tab <- age_tab[1:2000,]

# #############################################################################################################################
#### Adding training samples => 4k, 22 apr
num.add <- 4000

part_list2 <- read.csv('/well/nichols/users/qcv214/bnn2/add_1_part_id_use_final.txt')$V1 #4258
part_list2.exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list2,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')) #4257, only one person is missing
part_use2 <-part_list2[part_list2.exist_vbm]
part_use2 <- part_use2[1:num.add]
###
age_tab2<-as.data.frame(matrix(,nrow = length(part_use2),ncol = 2)) #id, age, number of masked voxels
colnames(age_tab2)[1:2]<-c('id','age')
age_tab2$id<-part_use2
for(i in 1:length(part_use2)){
  age_tab2$age[i]<-agetab$X21003.2.0[agetab$eid_8107==sub(".", "",age_tab2$id[i])]
}

age_tab<- rbind(age_tab,age_tab2)
#############################################################################################################################
n.train <- nrow(age_tab)

##Get data 

list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
sub.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
# sub.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz'))

colnames(sub.dat) <- as.character(1:ncol(sub.dat))
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab.test$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
sub.dat.test <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
# sub.dat.test <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/bnn2/res3/res3mask.nii.gz'))

colnames(sub.dat.test) <- colnames(sub.dat)

print("ridge")
#Ridge

set.seed(4)
# fit.ridge <- cv.glmnet(sub.dat,age_tab$age, alpha=0)
# beta <- coef(fit.ridge)
# beta_no_int <- beta[-1,]
# rank_beta.ridge <- beta_no_int[order(abs(beta_no_int), decreasing=TRUE)]
# num.vox.vec <- (1:100)*10
# 
# long.rmse <- function(ranked_coef, num.vox.vec){
#   train <- vector(mode = 'numeric')
#   test <- vector(mode = 'numeric')
#   for(i in num.vox.vec){
#     var.sel<- names(ranked_coef[1:i])
#     fit.ridge.pca <- cv.glmnet(sub.dat[,var.sel],age_tab$age, alpha=0)
#     beta <- coef(fit.ridge.pca)
#     train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat[,var.sel])%*%beta))^2)))
#     test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test[,var.sel])%*%beta))^2)))
#   }
#   out <- list()
#   out$train <- train
#   out$test <- test
#   return(out)
# }

# res.ridge <- long.rmse(rank_beta.ridge, num.vox.vec)


theta.range <- 10^seq(-4,3,1)


#Robust PCA

#cv but with selected columns only
varsel.cv <- function(lambda, no.variable = 1000,theta.range. = theta.range, varsel){ #Changed from -9 to -8, to -7 for 2k
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(theta in theta.range.){
    #doing PCA
    Omeg <- solve(sub.dat[,varsel]%*%lambda%*%t(sub.dat) + (theta)*diag(nrow(sub.dat[,varsel]))) #4x4 
    pre_beta_pms <- lambda%*%t(sub.dat[,varsel])%*%Omeg #4 x 75
    beta_pms <- pre_beta_pms%*%age_tab$age
    rownames(beta_pms) <- colnames(sub.dat[,varsel])
    rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
    
    #Doing prediction
    var.sel<- names(rank_beta_pms[1:no.variable])
    fit.ridge <- cv.glmnet(sub.dat[,varsel][,var.sel],age_tab$age, alpha=0)
    beta <- coef(fit.ridge, s = "lambda.min")
    train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat[,varsel][,var.sel])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test[,varsel][,var.sel])%*%beta))^2)))
  }
  out <- rbind(theta.range,train,test)
  return(out)
}

# print("tune robust pca")

# pca <- PcaHubert(sub.dat,kmax=n.train) 
# lambda. <- pca$loadings %*% t(pca$loadings) 
# 
# robpca.cv <- simple.cv(lambda =lambda.,no.variable = 1000,theta.range. = theta.range )

# print("tune pca")
# pca <-  prcomp(sub.dat)
# lambda. <- pca$rotation%*% t(pca$rotation)
# pca.cv <- simple.cv(lambda.) #theta = 10 is optimal
# 
# print("tune PLS")
# pls_model <- plsr(age_tab$age ~ sub.dat)
# lambda. <- loadings(pls_model) %*% t(loadings(pls_model)) #15847 x 15847
# pls.cv <- simple.cv(lambda.) #theta = 10 is optimal

print("tune SPCA")

cv_model <- cv.glmnet(sub.dat, age_tab$age, alpha = 0, nfolds = 10)
# Best lambda from cross-validation
coefficients <- coef(cv_model, s = "lambda.min")  # Extract coefficients at best lambda
weights <- abs(coefficients[-1,])  # Exclude intercept from weights (first row of coefficients)
X_weighted <- scale(sub.dat) * sqrt(weights) #there are 192 NAs in scale(sub.dat) but i dont know where *I believe it's to do with the column being a constant
#Question is do I rank the voxels by col mean or col var. I am guessing it's col var
colvar <- apply(X_weighted,MARGIN = 2, var)
rank_var.ridge <- colvar[order(colvar, decreasing=TRUE)]
var.sel<- names(rank_var.ridge[1:round(length(colvar)*0.3)]) #take only 30% of voxel
X_weighted_sel <- X_weighted[,var.sel]


#####
# Compute the MCD covariance matrix
mcd_result <- covMcd(X_weighted_sel)
mcd_cov <- mcd_result$cov
# Perform eigen decomposition on the covariance matrix
eig_decomp <- eigen(mcd_cov)
# Extract the eigenvectors
lambda. <- eig_decomp$vectors
#####


spca.cv <- varsel.cv(lambda.,varsel = var.sel) #theta = 10 is optimal


#below is practically long.rmse
print("making predictions")
long.rmse.with.lambda.theta.varsel.ridge <- function(lambda,theta,num.vox.vec,varsel){
  Omeg <- solve(sub.dat[,varsel]%*%lambda%*%t(sub.dat[,varsel]) + (theta)*diag(nrow(sub.dat[,varsel]))) #4x4 
  pre_beta_pms <- lambda%*%t(sub.dat[,varsel])%*%Omeg #4 x 75
  beta_pms <- pre_beta_pms%*%age_tab$age
  rownames(beta_pms) <- colnames(sub.dat[,varsel])
  rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(i in num.vox.vec){
    print(i)
    var.sel<- names(rank_beta_pms[1:i])
    #ridge prediction
    fit.ridge <- cv.glmnet(sub.dat[,varsel][,var.sel],age_tab$age, alpha=0)
    beta <- coef(fit.ridge, s = "lambda.min")
    train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat[,varsel][,var.sel])%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test[,varsel][,var.sel])%*%beta))^2)))
  }
  out <- list()
  out$train <- train
  out$test <- test
  return(out)
}


num.vox.vec <- (1:100)*100

# pca <- PcaHubert(sub.dat,kmax=100) 
# print("rmse robpca")
# 
# pca <- PcaHubert(sub.dat,kmax=n.train) 
# lambda. <- pca$loadings %*% t(pca$loadings) 
# print(paste0("dim ROBPCA: ", dim(pca$loadings)))
# res.lambda.ropca <- long.rmse.with.lambda.theta.ridge(lambda., robpca.cv[1,which.min(robpca.cv[3,])], num.vox.vec)
# 
# print("rmse pca")
# pca <-  prcomp(sub.dat)
# lambda. <- pca$rotation%*% t(pca$rotation)
# print(paste0("dim PCA: ", dim(pca$rotation)))
# res.lambda.pca <- long.rmse.with.lambda.theta.ridge(lambda., pca.cv[1,which.min(pca.cv[3,])], num.vox.vec)
# 
# 
# print("rmse PLS")
# pls_model <- plsr(age_tab$age ~ sub.dat)
# lambda. <- loadings(pls_model) %*% t(loadings(pls_model)) #15847 x 15847
# res.lambda.pls <- long.rmse.with.lambda.theta.ridge(lambda., pls.cv[1,which.min(pls.cv[3,])], num.vox.vec)
# 
# print("rmse SPCA")
# 
# cv_model <- cv.glmnet(sub.dat, age_tab$age, alpha = 0, nfolds = 10)
# # Best lambda from cross-validation
# coefficients <- coef(cv_model, s = "lambda.min")  # Extract coefficients at best lambda
# weights <- abs(coefficients[-1,])  # Exclude intercept from weights (first row of coefficients)
# X_weighted <- scale(sub.dat) * sqrt(weights)
# # Perform PCA on the weighted matrix
# pca<- prcomp(X_weighted, center = TRUE, scale. = FALSE)
# lambda. <- pca$rotation%*% t(pca$rotation)
res.lambda.spca <- long.rmse.with.lambda.theta.varsel.ridge(lambda., spca.cv[1,which.min(spca.cv[3,])], num.vox.vec, varsel = var.sel)




#Need to save saliency and


out <- rbind(res.lambda.spca$train,res.lambda.spca$test)
# print(paste0("ROBUST optimal Theta: ", robpca.cv[1,which.min(robpca.cv[3,])]))
# print(paste0("PCA optimal Theta: ", pca.cv[1,which.min(pca.cv[3,])]))
# print(paste0("pls optimal Theta: ",pls.cv[1,which.min(pls.cv[3,])]))
print(paste0("ridge-SRPCA optimal Theta: ", spca.cv[1,which.min(spca.cv[3,])]))
write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/june9_cv_pms_ridgeSMCDPCA.csv'), row.names = FALSE)




