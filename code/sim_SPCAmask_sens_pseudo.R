#Aug 30, introduce a double cv. (one during screening, one during pred which shouldn't matter), but more importantly, correctly cv during screening for ridge

#2 sep, psudo HOLP

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

res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')

mask.rank <- c(fast_read_imgs_mask('/well/nichols/users/qcv214/pms2/viz/sim/aug21_500_age_rank_SPCA90_smooth.nii.gz','/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))

mask.signal <- order(mask.rank,decreasing = TRUE)[1:1600]

##################################################################
signal.strs <- c(0,0.5,1,2,3,5)
signal.strs.diff <- c(0,diff(signal.strs))

for(q in (1:length(signal.strs))){
  
  signal.str <- signal.strs[q]
  
  print("here 4")
  ####Calling mmand
  
  
  sub.dat[,mask.signal] <- sub.dat[,mask.signal] + signal.strs.diff[q]
  
  sub.dat.test[,mask.signal] <- sub.dat.test[,mask.signal] + signal.strs.diff[q]
  
  # mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  # mask.temp[mask.temp!=0] <- colMeans(sub.dat)
  # mask.temp@datatype = 16
  # mask.temp@bitpix = 32
  # writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug30_500_SPCAmask_str',signal.str,'_meantrainingdata'))
  # 
  # mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  # mask.temp[mask.temp!=0] <- colMeans(sub.dat.test)
  # mask.temp@datatype = 16
  # mask.temp@bitpix = 32
  # writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug30_500_SPCAmask_str',signal.str,'_meantestdata'))
  
  
  signal.sum.func <- function(dat,indices){
    return(sum(dat[indices]))
  }
  
  res.var <- apply(sub.dat, MARGIN = 1, FUN = signal.sum.func, indices = mask.signal)
  res.var.test <- apply(sub.dat.test, MARGIN = 1, FUN = signal.sum.func, indices = mask.signal)
  
  num.vox.vec <- (1:40)*100
  
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
  
  
  # num.vox.vec <- (1:50)*200
  
  # pca <- PcaHubert(sub.dat,kmax=100) 
  print("rmse robpca")
  
  # pca <- PcaHubert(sub.dat,kmax=n.train) 
  # lambda. <- pca$loadings %*% t(pca$loadings) 
  # print(paste0("dim ROBPCA: ", dim(pca$loadings)))
  # res.lambda.ropca <- long.rmse.with.lambda.theta.ridge(lambda., robpca.cv[1,which.min(robpca.cv[3,])], num.vox.vec)
  print("tune pca")
  
  print("tune pca 90")
  lambda.80 <- diag(ncol(sub.dat)) ##### identify matrix.
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
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug30_500_SPCAmask_str',signal.str,'_pseuholp'))
  ###
  
  var.sel <- (order(abs(beta_pms), decreasing=TRUE))
  
  mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  mask.temp[mask.temp!=0][var.sel] <- c(ncol(sub.dat):1)
  mask.temp@datatype = 16
  mask.temp@bitpix = 32
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug30_500_SPCAmask_rank_str',signal.str,'_pseuholp'))
  
  
  
}