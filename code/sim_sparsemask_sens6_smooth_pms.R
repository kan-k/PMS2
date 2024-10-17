#Aug 30, introduce a double cv. (one during screening, one during pred which shouldn't matter), but more importantly, correctly cv during screening for ridge
# install.packages("feather")
# library(rrcov)
#Sep 26, use true beta derived from `gen_age_HOLP_binmask.R` and data, as assessed, gen from U[0,1]

#Oct 3, I changed it such that it doesn't use the full subbrain signal, only first 300. And increased number of subject to 1000 for training, and reduce number of evaluating voxels

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
part_use<-part_use[1:1500,] #only take 4k

agetab<-read.table(file = '/well/nichols/projects/UKB/SMS/ukb_latest-Age.tsv', sep = '\t', header = TRUE)
age_tab<-as.data.frame(matrix(,nrow = length(part_use$V1),ncol = 2)) #id, age, number of masked voxels
colnames(age_tab)[1:2]<-c('id','age')
age_tab$id<-part_use$V1
age_tab.test <- age_tab[1001:1500,]

# age_tab <- age_tab[1:2000,]
age_tab <- age_tab[1:1000,]

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
mask.signal <- which(mask.index)[seq(1,1500,5)] #length is 300
signal_types <- "sparsemask"


######## add for loop here
# Define the accuracy calculation function
accuracy_cal <- function(rank.vec, mask.signal) {
  n <- length(mask.signal)
  top_n <- rank.vec[1:n]
  TP <- length(intersect(top_n, mask.signal))
  overlap <- round(TP * 100 / n, 3)
  FP <- length(setdiff(top_n, mask.signal))
  total_non_signal <- length(rank.vec) - n
  FPR <- round(FP * 100 / total_non_signal, 3)
  FDR <- round(FP * 100 / (FP + TP), 3)
  
  return(list(
    overlap = overlap,
    FPR = FPR,
    FDR = FDR
  ))
}

# Parameters for simulation
n_iterations <- 3 #was 10

# Initialize a data frame to store the results
results <- data.frame(
  SignalType = character(),
  Method = character(),
  SignalStrength = numeric(),
  Iteration = numeric(),
  Overlap = numeric(),
  FPR = numeric(),
  FDR = numeric(),
  stringsAsFactors = FALSE
)


for (iter in 1:n_iterations) {
  for (a in signal_types) {
    ########
    
    # set.seed(4)
    sub.dat[,mask.signal] <- matrix(runif(nrow(sub.dat)*length(mask.signal),0,1) , nrow = nrow(sub.dat), ncol = length(mask.signal))
    # set.seed(4)
    sub.dat.test[,mask.signal] <-  matrix(runif(nrow(sub.dat.test)*length(mask.signal),0,1) , nrow = nrow(sub.dat.test), ncol = length(mask.signal))
    
    ########
    mask.beta <- c(fast_read_imgs_mask('/well/nichols/users/qcv214/pms2/viz/sep25_age_HOLP_binmask_fit.nii.gz','/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
    res.var <- sub.dat[,mask.signal] %*% mask.beta[mask.signal]
    res.var.test <- sub.dat.test[,mask.signal] %*% mask.beta[mask.signal]
    var.pred <- sum((mask.beta[mask.signal])^2)/12 #theoretical variance => from 0.1^2/12
    ########
    
    rsquares <- c(1,0.9,0.5)
    # noise.strs <- c(0, var.pred/0.9 - var.pred, var.pred/0.5 - var.pred) #R2 = 1, 0.9, 0.5
    noise.strs <- var.pred/rsquares - var.pred
    noise.strs.diff <- c(0,diff(noise.strs))
    
    for(q in (1:length(noise.strs))){
      
      
      
      noise.str <- noise.strs[q]
      rsquare <- rsquares[q]
      # set.seed(4)
      res.var <- res.var + rnorm(length(res.var), 0, sqrt(noise.strs.diff[q]))
      res.var.test <- res.var.test + rnorm(length(res.var.test), 0, sqrt(noise.strs.diff[q]))
      
      
      ################################################################ pca smooth 3 ######################################################################
      shrinkage.param <- 0.1
      
      smooth.params <- c(0.25,0.5,1,2)
      
      for(smooth.param in smooth.params){
        
        print("#####################################")
        print(paste0("This is interation: ",iter,"| rsquare level: ",rsquare,"| smoothing param: ",smooth.param))
        
        smooth_1d_image <- function(x){
          #turn 1d image into 3d
          gp.mask.hs <- res3.mask
          gp.mask.hs[gp.mask.hs!=0] <- x
          #smoothen
          gp.mask.hs <- gaussianSmooth(gp.mask.hs, c(smooth.param,smooth.param,smooth.param))
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
        
        
        #PMS
        img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
        nb <- find_brain_image_neighbors(img1, res3.mask, radius=1)
        
        n.train <- nrow(sub.dat.s)
        n.test <- nrow(sub.dat.s.test)
        rho_high <- rep(-log(0.9)/4,length=ncol(sub.dat.s))
        #pms high
        out.pms.hi<-fast_PMS_local_spatial(x=as.matrix(sub.dat.s), y = res.var, coords=nb$maskcoords,neighbors=nb$mask_img_nb,num_neighbors=nb$num_neighbors, rho = rho_high)$pms_select
        #holp
        out.holp<-fast_PMS_cpp(x=as.matrix(sub.dat.s), y = res.var, theta = 0)$pms_select
        
        metrics <- accuracy_cal(rank.vec = out.pms.hi, mask.signal = mask.signal)
        results <- rbind(results, data.frame(SignalType = a, Method = paste0("pms_smooth",smooth.param), SignalStrength = rsquare, Iteration = iter,Overlap = metrics$overlap,FPR = metrics$FPR,FDR = metrics$FDR))
        metrics <- accuracy_cal(rank.vec = out.holp, mask.signal = mask.signal)
        results <- rbind(results, data.frame(SignalType = a, Method = paste0("holp_smooth",smooth.param), SignalStrength = rsquare, Iteration = iter,Overlap = metrics$overlap,FPR = metrics$FPR,FDR = metrics$FDR))
        
      }
      
    }
  }
}

write.csv(results, '/well/nichols/users/qcv214/pms2/pile/sim_oct11_sparsemask_smooth_pms.csv', row.names = FALSE)
