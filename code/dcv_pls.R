#Use res3mask with 150k voxels

#19 Apr: use *Centre mask* and 1000 subjects.


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

#############################################################################################################################
### Adding training samples => 4k, 22 apr
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

# print("ridge")
#Ridge

# set.seed(4)
# fit.ridge <- cv.glmnet(sub.dat,age_tab$age, alpha=0)
# beta <- coef(fit.ridge)
# beta_no_int <- beta[-1,]
# rank_beta.ridge <- beta_no_int[order(abs(beta_no_int), decreasing=TRUE)]
# num.vox.vec <- (1:100)*100
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
# 
# res.ridge <- long.rmse(rank_beta.ridge, num.vox.vec)

#Robust PCA

theta.range <- 10^seq(-4,3,1)

for.double.cv <- function(no.variable = 1000,lambda,theta1, theta2){ #Changed from -9 to -8, to -7 for 2k
  #doing PCA
  Omeg <- solve(sub.dat%*%lambda%*%t(sub.dat) + (theta1)*diag(nrow(sub.dat))) #4x4 
  pre_beta_pms <- lambda%*%t(sub.dat)%*%Omeg #4 x 75
  beta_pms <- pre_beta_pms%*%age_tab$age
  rownames(beta_pms) <- colnames(sub.dat)
  rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
  
  #Doing prediction
  var.sel<- names(rank_beta_pms[1:no.variable])
  sub.dat.sub <- sub.dat
  sub.dat.test.sub <- sub.dat.test
  
  mask.out <- as.numeric(setdiff(colnames(sub.dat),var.sel))
  #print(head(mask.out))
  
  sub.dat.sub <- sub.dat.sub[,-c(mask.out)] 
  sub.dat.test.sub<- sub.dat.test.sub[,-c(mask.out)] 
  
  lambda.sub <- lambda[-c(mask.out),-c(mask.out)]
  
  Omeg <- solve(sub.dat.sub%*%lambda.sub%*%t(sub.dat.sub) + (theta2)*diag(nrow(sub.dat.sub)))
  beta_pms <- lambda.sub%*%t(sub.dat.sub)%*%Omeg%*%age_tab$age
  intercept <- mean(age_tab$age) - colMeans(sub.dat.sub)%*%beta_pms 
  
  beta <- c(intercept,beta_pms)
  train <- sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat.sub)%*%beta))^2))
  test <- sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test.sub)%*%beta))^2))
  out <- c(train,test)
  return(out)
}


print("tune PLS")
pls_model <- plsr(age_tab$age ~ sub.dat)
scores <- scores(pls_model)
lambda <- loadings(pls_model) %*% t(loadings(pls_model)) #15847 x 15847

out <- matrix(NA, nrow = length(theta.range), ncol = length(theta.range))
for(i in 1:length(theta.range)){
  for(j in 1:length(theta.range)){
    out[i,j] <- for.double.cv(no.variable = 1000, lambda = lambda ,theta1 = theta.range[i], theta2 = theta.range[j])[2]
  }
}
min_val_index <- which(out == min(out), arr.ind = TRUE)
robpca.best.theta1 <- theta.range[min_val_index[1]]
robpca.best.theta2 <- theta.range[min_val_index[2]]
print(paste0("ROBUST optimal screening Theta: ",robpca.best.theta1))
print(paste0("ROBUST optimal predict Theta: ",robpca.best.theta2))


# print("tune pca")
# pca <-  prcomp(sub.dat)
# lambda <- pca$rotation%*% t(pca$rotation)
# 
# out <- matrix(NA, nrow = length(theta.range), ncol = length(theta.range))
# for(i in 1:length(theta.range)){
#   for(j in 1:length(theta.range)){
#     out[i,j] <- for.double.cv(no.variable = 1000, lambda = lambda ,theta1 = theta.range[i], theta2 = theta.range[j])[2]
#   }
# }
# min_val_index <- which(out == min(out), arr.ind = TRUE)
# pca.best.theta1 <- theta.range[min_val_index[1]]
# pca.best.theta2 <- theta.range[min_val_index[2]]
# print(paste0("PCA optimal screening Theta: ",pca.best.theta1))
# print(paste0("PCA optimal predict Theta: ",pca.best.theta2))

print("rmse robust pca")
long.rmse.with.lambda.theta <- function(lambda,theta1, theta2,num.vox.vec){
  Omeg <- solve(sub.dat%*%lambda%*%t(sub.dat) + (theta1)*diag(nrow(sub.dat))) #4x4 
  pre_beta_pms <- lambda%*%t(sub.dat)%*%Omeg #4 x 75
  beta_pms <- pre_beta_pms%*%age_tab$age
  rownames(beta_pms) <- colnames(sub.dat)
  rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
  train <- vector(mode = 'numeric')
  test <- vector(mode = 'numeric')
  for(i in num.vox.vec){
    var.sel<- names(rank_beta_pms[1:i])
    sub.dat.sub <- sub.dat
    sub.dat.test.sub <- sub.dat.test
    
    mask.out <- as.numeric(setdiff(colnames(sub.dat),var.sel))
    
    sub.dat.sub <- sub.dat.sub[,-c(mask.out)] 
    sub.dat.test.sub<- sub.dat.test.sub[,-c(mask.out)] 
    
    lambda.sub <- lambda[-c(mask.out),-c(mask.out)]
    
    Omeg <- solve(sub.dat.sub%*%lambda.sub%*%t(sub.dat.sub) + (theta2)*diag(nrow(sub.dat.sub)))
    beta_pms <- lambda.sub%*%t(sub.dat.sub)%*%Omeg%*%age_tab$age
    intercept <- mean(age_tab$age) - colMeans(sub.dat.sub)%*%beta_pms 
    
    beta <- c(intercept,beta_pms)
    train <- c(train,sqrt(mean((as.numeric(age_tab$age - cbind(1,sub.dat.sub)%*%beta))^2)))
    test <- c(test,sqrt(mean((as.numeric(age_tab.test$age - cbind(1,sub.dat.test.sub)%*%beta))^2)))
  }
  out <- list()
  out$train <- train
  out$test <- test
  return(out)
}

num.vox.vec <- (1:100)*100

# pca <- PcaHubert(sub.dat,kmax=100) 
pls_model <- plsr(age_tab$age ~ sub.dat)
scores <- scores(pls_model)
lambda <- loadings(pls_model) %*% t(loadings(pls_model)) #15847 x 15847
print(paste0("dim ROBPCA: ", dim(loadings(pls_model))))
# pca <-  prcomp(sub.dat)
# lambda.normal <- pca$rotation%*% t(pca$rotation)
# print(paste0("dim PCA: ", dim(pca$rotation)))

res.lambda.ropca <- long.rmse.with.lambda.theta(lambda, robpca.best.theta1,robpca.best.theta2, num.vox.vec)

# print("rmse pca")

# res.lambda.pca <- long.rmse.with.lambda.theta(lambda.normal, pca.best.theta1,pca.best.theta2, num.vox.vec)


#Need to save saliency and


# out <- rbind(res.lambda.ropca$train,res.lambda.pca$train,res.ridge$train,res.lambda.ropca$test,res.lambda.pca$test,res.ridge$test)
out <- rbind(res.lambda.ropca$train,res.lambda.ropca$test)


write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/may9_dcv_pls.csv'), row.names = FALSE)



####Viz
print("VIZ -------- ")

print("Robust PCA Viz: fitting model")
#Robust PCA
set.seed(4)
# pca <- PcaHubert(sub.dat,kmax=100) #it only picks 10 PC ####Note that I need to change kmax to greater than 10
pls_model <- plsr(age_tab$age ~ sub.dat)
scores <- scores(pls_model)
lambda <- loadings(pls_model) %*% t(loadings(pls_model)) #15847 x 15847
Omeg <- solve(sub.dat%*%lambda%*%t(sub.dat) + (robpca.best.theta1)*diag(nrow(sub.dat))) #4x4 
pre_beta_pms <- lambda%*%t(sub.dat)%*%Omeg #4 x 75
beta_pms <- pre_beta_pms%*%age_tab$age
# rownames(beta_pms) <- colnames(sub.dat)
# rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
print("Robust PCA Viz: Brain plot")
# #Full ROBPCA
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- abs(c(beta_pms))
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/may9_pls_pms'))

# #PCA 
# print("PCA Viz: fitting model")
# 
# set.seed(4)
# pca.normal <- prcomp(sub.dat)
# #lambda.normal <- pca.normal$rotation[,ind.touse] %*% t(pca.normal$rotation[,ind.touse])
# lambda.normal <- pca.normal$rotation%*% t(pca.normal$rotation)
# Omeg <- solve(sub.dat%*%lambda.normal%*%t(sub.dat) + (pca.best.theta1)*diag(nrow(sub.dat))) #4x4 
# pre_beta_pms.normal <- lambda.normal%*%t(sub.dat)%*%Omeg #4 x 75
# beta_pms.normal <- pre_beta_pms.normal %*%age_tab$age
# # rownames(beta_pms.normal) <- colnames(sub.dat)
# # rank_beta_pms.normal <- beta_pms.normal[order(abs(beta_pms.normal), decreasing=TRUE),]
# print("PCA Viz: Brain plot")
# # #PCA
# #Full PCA
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
# mask.temp[mask.temp!=0] <- abs(c(beta_pms.normal))
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/apr23_pca_pms'))
# # 
# # #Ridge
# # #Full Ridge
# print("Ridge Viz: Brain plot")
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
# mask.temp[mask.temp!=0] <- abs(c(beta_no_int))
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/apr23_ridge'))

