#Aug 30, introduce a double cv. (one during screening, one during pred which shouldn't matter), but more importantly, correctly cv during screening for ridge
# install.packages("feather")
# library(rrcov)
#Sep 2, instead of adding signals to raw data, i'll replace signals

if (!require("pacman")) {install.packages("pacman");library(pacman)}
p_load(BayesGPfit)
p_load(Matrix)
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
sub.dat[,mask.signal] <- matrix(runif(nrow(sub.dat)*length(mask.signal),0,0.1) , nrow = nrow(sub.dat), ncol = length(mask.signal))
set.seed(4)
sub.dat.test[,mask.signal] <-  matrix(runif(nrow(sub.dat.test)*length(mask.signal),0,0.1) , nrow = nrow(sub.dat.test), ncol = length(mask.signal))

#Assume beta = 1, so we don't specify anything. and vartiance is simple, which corresponds to y being sum of active voxels
signal.sum.func <- function(dat,indices){
  return(sum(dat[indices]))
}

res.var <- apply(sub.dat, MARGIN = 1, FUN = signal.sum.func, indices = mask.signal)
res.var.test <- apply(sub.dat.test, MARGIN = 1, FUN = signal.sum.func, indices = mask.signal)

# var.pred <- var(res.var) #empirical variance
var.pred <- length(mask.signal)/1200 #theoretical variance => from 0.1^2/12

rsquares <- c(1,0.9,0.5)
# noise.strs <- c(0, var.pred/0.9 - var.pred, var.pred/0.5 - var.pred) #R2 = 1, 0.9, 0.5
noise.strs <- var.pred/rsquares - var.pred
noise.strs.diff <- c(0,diff(noise.strs))

for(q in (1:length(noise.strs))){
  
  noise.str <- noise.strs[q]
  rsquare <- rsquares[q]
  set.seed(4)
  res.var <- res.var + rnorm(length(res.var), 0, sqrt(noise.strs.diff[q]))
  res.var.test <- res.var.test + rnorm(length(res.var.test), 0, sqrt(noise.strs.diff[q]))
  
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
      
      var.sel<- pms_ind[1:i]
      #ridge prediction
      # fit.ridge <- cv.glmnet(sub.dat[,var.sel],res.var, alpha=0)
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
  
  res.pms.hi <- long.rmse.pms(out.pms.hi, num.vox.vec)
  res.pms.holp <- long.rmse.pms(out.holp, num.vox.vec)
  res.pms.boot <- long.rmse.pms(out.pms.boot, num.vox.vec)
  
  out <- rbind(res.pms.hi$train,res.pms.holp$train ,res.pms.boot$train, 
               res.pms.hi$test,res.pms.holp$test,res.pms.boot$test)
  
  write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/sim_sep20_test_500_binmask_str',rsquare,'_nbpms.csv'), row.names = FALSE)
  
  #plot out.pms.hi
  mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  mask.temp[mask.temp!=0][out.pms.hi] <- c(1:ncol(sub.dat))
  mask.temp@datatype = 16
  mask.temp@bitpix = 32
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/sep20_test_500_binmask_rank_str',rsquare,'_pmshi'))
  
  #plot out.pms.holp
  mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
  mask.temp[mask.temp!=0][out.holp] <- c(1:ncol(sub.dat))
  mask.temp@datatype = 16
  mask.temp@bitpix = 32
  writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/sep20_test_500_binmask_rank_str',rsquare,'_holp'))

  
}
