
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
#   res.var[i]<-agetab$X21003.2.0[agetab$eid_8107==sub(".", "",age_tab$id[i])]
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

signal.sum.func <- function(dat,indices){
  return(sum(dat[indices]))
}

res.var <- apply(sub.dat, MARGIN = 1, FUN = signal.sum.func, indices = mask.signal)
res.var.test <- apply(sub.dat.test, MARGIN = 1, FUN = signal.sum.func, indices = mask.signal)


print("here 4")
####Calling mmand
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

write.csv(out,paste0( '/well/nichols/users/qcv214/pms2/pile/sim_aug14_500_str',signal.str,'_nbpms.csv'), row.names = FALSE)

#plot out.pms.hi
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0][out.pms.hi] <- c(ncol(sub.dat):1)/ncol(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug14_500_str',signal.str,'_pmshi'))

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- 1
mask.temp[mask.temp!=0][head(out.pms.hi,1867)] <- 2
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug14_500_top_str',signal.str,'_pmshi'))

#plot out.pms.holp
mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0][out.holp] <- c(ncol(sub.dat):1)/ncol(sub.dat)
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug14_500_str',signal.str,'_holp'))

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp!=0] <- 1
mask.temp[mask.temp!=0][head(out.holp,1867)] <- 2
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/sim/aug14_500_top_str',signal.str,'_holp'))

#plot out.pms.boot
# mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
# mask.temp[mask.temp!=0][out.pms.boot] <- c(ncol(sub.dat):1)/ncol(sub.dat)
# mask.temp@datatype = 16
# mask.temp@bitpix = 32
# writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/july26_pms_boot'))




