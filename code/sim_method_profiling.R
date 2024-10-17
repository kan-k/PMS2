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

# install.packages("/well/nichols/users/qcv214/pms2/package/mmand_1.6.2.tar.gz", repos = NULL, type = "source")
library(mmand)

##########
################################################################ pca no-smooth ######################################################################
print("load data")
print(Sys.time())
start.time <- Sys.time()

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
mask.signal <- which(mask.index)[1:300] #length is 300
signal_types <- "binmask"

time.taken <- Sys.time() - start.time
cat("load all data completed in: ", time.taken)


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
methods <- c("PCA90", "SPCA90", "PCA90_smooth", "SPCA90_smooth", "ridge", "ridge_smooth", "pmshi", "holp", "tmap",
             "PCA90_smooth_proj", "PCA90_smooth_cov", "SPCA90_smooth_proj", "SPCA90_smooth_cov")
n_iterations <- 1

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
    
    rsquares <- c(1)
    # noise.strs <- c(0, var.pred/0.9 - var.pred, var.pred/0.5 - var.pred) #R2 = 1, 0.9, 0.5
    noise.strs <- var.pred/rsquares - var.pred
    noise.strs.diff <- c(0,diff(noise.strs))
    
    for(q in (1:length(noise.strs))){
      
      noise.str <- noise.strs[q]
      rsquare <- rsquares[q]
      # set.seed(4)
      res.var <- res.var + rnorm(length(res.var), 0, sqrt(noise.strs.diff[q]))
      res.var.test <- res.var.test + rnorm(length(res.var.test), 0, sqrt(noise.strs.diff[q]))
      
      
      
      # print("ridge")
      # #Ridge
      # 
      #     print(Sys.time())
      #     start.time <- Sys.time()
      # 
      # # set.seed(4)
      # fit.ridge <- cv.glmnet(sub.dat,res.var, alpha=0)
      # beta <- coef(fit.ridge, s = "lambda.min")
      # beta_no_int <- beta[-1,]
      # rank_beta.ridge <- beta_no_int[order(abs(beta_no_int), decreasing=TRUE)]
      # num.vox.vec <- (1:3)*100
      # 
      # 
      # var.sel <- (order(abs(beta_no_int), decreasing=TRUE))
      # 
      #     time.taken <- Sys.time() - start.time
      #     cat("run ridge CV completed in: ", time.taken)
      #     
      #     
      #     print(Sys.time())
      #     start.time <- Sys.time()
      #     
      #     # set.seed(4)
      #     fit.ridge <- glmnet(sub.dat,res.var, alpha=0, lambda = 1e-3)
      #     beta <- coef(fit.ridge)
      #     beta_no_int <- beta[-1,]
      #     rank_beta.ridge <- beta_no_int[order(abs(beta_no_int), decreasing=TRUE)]
      #     num.vox.vec <- (1:3)*100
      #     var.sel <- (order(abs(beta_no_int), decreasing=TRUE))
      #     
      #     time.taken <- Sys.time() - start.time
      #     cat("run ridge no-CV completed in: ", time.taken)
      # 
      # 
      # # theta.range <- 10^seq(-4,3,1)
      # 
      # theta.range <- 10^seq(-1,3,1)
      # 
      # 
      # #Robust PCA
      # 
      # 
      # simple.cv <- function(lambda, no.variable = 1000,theta.range. = theta.range){ #Changed from -9 to -8, to -7 for 2k
      #   train <- vector(mode = 'numeric')
      #   test <- vector(mode = 'numeric')
      #   for(theta in theta.range.){
      #     #doing PCA
      #     
      #     Omeg <- solve(sub.dat%*%lambda%*%t(sub.dat) + (theta)*diag(nrow(sub.dat))) #4x4 
      #     
      #     pre_beta_pms <- lambda%*%t(sub.dat)%*%Omeg #4 x 75
      #     
      #     beta_pms <- pre_beta_pms%*%res.var
      #     
      #     rownames(beta_pms) <- colnames(sub.dat)
      #     rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
      #     
      #     #Doing prediction
      #     var.sel<- names(rank_beta_pms[1:no.variable])
      #     
      #     fit.ridge <- cv.glmnet(sub.dat[,var.sel],res.var, alpha=0)
      #     beta <- coef(fit.ridge, s = "lambda.min")
      #     train <- c(train,sqrt(mean((as.numeric(res.var - cbind(1,sub.dat[,var.sel])%*%beta))^2)))
      #     test <- c(test,sqrt(mean((as.numeric(res.var.test - cbind(1,sub.dat.test[,var.sel])%*%beta))^2)))
      #   }
      #   out <- rbind(theta.range,train,test)
      #   return(out)
      # }
      # 
      # 
      # 
      # # num.vox.vec <- (1:50)*200
      # 
      # # pca <- PcaHubert(sub.dat,kmax=100) 
      # print("rmse robpca")
      # 
      # # pca <- PcaHubert(sub.dat,kmax=n.train) 
      # # lambda. <- pca$loadings %*% t(pca$loadings) 
      # # print(paste0("dim ROBPCA: ", dim(pca$loadings)))
      # # res.lambda.ropca <- long.rmse.with.lambda.theta.ridge(lambda., robpca.cv[1,which.min(robpca.cv[3,])], num.vox.vec)
      # print("tune pca")
      # 
      # print(Sys.time())
      # start.time <- Sys.time()
      # 
      # pca <-  prcomp(sub.dat)
      # explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
      # # Calculate the cumulative variance explained
      # cumulative_variance <- cumsum(explained_variance)
      # # Determine the number of components needed to explain at least 90% of the variance
      # num_components.90 <- which(cumulative_variance >= 0.90)[1]
      # 
      # print("tune pca 90")
      # lambda.80 <- pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90])
      # #save col means
      # 
      # pca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
      # print("rmse pca 90")
      # 
      # #Plotting 
      # Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (pca.cv.90[1,which.min(pca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
      # pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
      # beta_pms <- pre_beta_pms%*%res.var
      # rownames(beta_pms) <- colnames(sub.dat)
      # rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
      # 
      # var.sel <- (order(abs(beta_pms), decreasing=TRUE))
      # 
      # time.taken <- Sys.time() - start.time
      # cat("run PCA completed in: ", time.taken)
      # 
      # print("tune SPCA")
      # 
      # 
      # print(Sys.time())
      # start.time <- Sys.time()
      # 
      # cv_model <- cv.glmnet(sub.dat, res.var, alpha = 0, nfolds = 10)
      # # Best lambda from cross-validation
      # coefficients <- coef(cv_model, s = "lambda.min")  # Extract coefficients at best lambda
      # weights <- abs(coefficients[-1,])  # Exclude intercept from weights (first row of coefficients)
      # X_weighted <- scale(sub.dat) * sqrt(weights)
      # 
      # time.taken <- Sys.time() - start.time
      # cat("run re-weighting part of SPCA completed in: ", time.taken)
      # # Perform PCA on the weighted matrix
      # ################################################################ PMS ######################################################################
      # 
      # 
      # res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
      # 
      # 
      # #PMS
      # img1 <- oro.nifti::readNIfTI(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_use[1,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
      # nb <- find_brain_image_neighbors(img1, res3.mask, radius=1)
      # 
      # n.train <- nrow(sub.dat)
      # n.test <- nrow(sub.dat.test)
      # rho_high <- rep(-log(0.9)/4,length=ncol(sub.dat))
      # 
      # num.vox.vec <- (1:3)*100
      # 
      # 
      # 
      # print("stage 3")
      # 
      # #pms high
      # print(Sys.time())
      # start.time <- Sys.time()
      # out.pms.hi<-fast_PMS_local_spatial(x=as.matrix(sub.dat), y = res.var, coords=nb$maskcoords,neighbors=nb$mask_img_nb,num_neighbors=nb$num_neighbors, rho = rho_high)$pms_select
      # #holp
      # time.taken <- Sys.time() - start.time
      # cat("PMS completed in: ", time.taken)
      # 
      # print(Sys.time())
      # start.time <- Sys.time()
      # out.holp<-fast_PMS_cpp(x=as.matrix(sub.dat), y = res.var, theta = 0)$pms_select
      # time.taken <- Sys.time() - start.time
      # cat("HOLP completed in: ", time.taken)
      # 
      # 
      # ################################################################ pca smooth 3 ######################################################################
      # shrinkage.param <- 0.1
      # 
      # print(Sys.time())
      # start.time <- Sys.time()
      # 
      # smooth_1d_image <- function(x){
      #   #turn 1d image into 3d
      #   gp.mask.hs <- res3.mask
      #   gp.mask.hs[gp.mask.hs!=0] <- x
      #   #smoothen
      #   gp.mask.hs <- gaussianSmooth(gp.mask.hs, c(1,1,1))
      #   #mask it again
      #   gp.mask.hs <- gp.mask.hs[res3.mask!=0]
      #   return(gp.mask.hs)
      # }
      # sub.dat.s <- t(apply(sub.dat,MARGIN = 1, smooth_1d_image))
      # sub.dat.s.test <- t(apply(sub.dat.test,MARGIN = 1, smooth_1d_image))
      # 
      # colnames(sub.dat.s) <- column.names
      # colnames(sub.dat.s.test) <- column.names
      # 
      # colnames(sub.dat) <- column.names
      # colnames(sub.dat.test) <- column.names
      # 
      # time.taken <- Sys.time() - start.time
      # cat("Smoothing images completed in: ", time.taken)
      # 
      # 
      # theta.range <- 10^seq(-1,3,1)
      # 
      # simple.cv <- function(lambda, no.variable = 1000,theta.range. = theta.range){ #Changed from -9 to -8, to -7 for 2k
      #   train <- vector(mode = 'numeric')
      #   test <- vector(mode = 'numeric')
      #   for(theta in theta.range.){
      #     #doing PCA
      #     
      #     Omeg <- solve(sub.dat.s%*%lambda%*%t(sub.dat.s) + (theta)*diag(nrow(sub.dat))) #4x4 
      #     
      #     pre_beta_pms <- lambda%*%t(sub.dat.s)%*%Omeg #4 x 75
      #     
      #     beta_pms <- pre_beta_pms%*%res.var
      #     
      #     rownames(beta_pms) <- colnames(sub.dat.s)
      #     rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
      #     
      #     #Doing prediction
      #     var.sel<- names(rank_beta_pms[1:no.variable])
      #     
      #     fit.ridge <- cv.glmnet(sub.dat[,var.sel],res.var, alpha=0)
      #     beta <- coef(fit.ridge, s = "lambda.min")
      #     train <- c(train,sqrt(mean((as.numeric(res.var - cbind(1,sub.dat[,var.sel])%*%beta))^2)))
      #     test <- c(test,sqrt(mean((as.numeric(res.var.test - cbind(1,sub.dat.test[,var.sel])%*%beta))^2)))
      #   }
      #   out <- rbind(theta.range,train,test)
      #   return(out)
      # }
      # 
      # #below is practically long.rmse
      # print("making predictions")
      # 
      # print("tune pca")
      # 
      # print(Sys.time())
      # start.time <- Sys.time()
      # 
      # pca <-  prcomp(sub.dat)
      # explained_variance <- pca$sdev^2 / sum(pca$sdev^2)
      # # Calculate the cumulative variance explained
      # cumulative_variance <- cumsum(explained_variance)
      # # Determine the number of components needed to explain at least 90% of the variance
      # num_components.90 <- which(cumulative_variance >= 0.90)[1]
      # 
      # print("tune pca 90")
      # lambda.80 <- pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90])
      # #save col means
      # 
      # pca.cv.90 <- simple.cv(lambda.80) #theta = 10 is optimal
      # print("rmse pca 90")
      # 
      # #Plotting 
      # Omeg <- solve(sub.dat.s%*%lambda.80%*%t(sub.dat.s) + (pca.cv.90[1,which.min(pca.cv.90[3,])])*diag(nrow(sub.dat))) #4x4
      # pre_beta_pms <- lambda.80%*%t(sub.dat.s)%*%Omeg #4 x 75
      # beta_pms <- pre_beta_pms%*%res.var
      # rownames(beta_pms) <- colnames(sub.dat)
      # rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
      # 
      # var.sel <- (order(abs(beta_pms), decreasing=TRUE))
      # time.taken <- Sys.time() - start.time
      # cat("PCA_smooth completed in: ", time.taken)
      # 
      # 
      # print(Sys.time())
      # start.time <- Sys.time()
      # lambda.80 <- shrinkage.param*(pca$rotation[, 1:num_components.90]%*% t(pca$rotation[, 1:num_components.90]))
      # diag(lambda.80) <- diag(lambda.80) + (1-shrinkage.param)
      # 
      # time.taken <- Sys.time() - start.time
      # cat("Projection shrinkage (given PCA) completed in: ", time.taken)
      # 
      # ###
      # print(Sys.time())
      # start.time <- Sys.time()
      # lambda.80 <- shrinkage.param*pca$rotation[, 1:num_components.90] %*% diag(pca$sdev[1:num_components.90]^2) %*% t(pca$rotation[, 1:num_components.90])
      # diag(lambda.80) <- diag(lambda.80) + (1-shrinkage.param)
      # time.taken <- Sys.time() - start.time
      # cat("Covariance shrinkage (given PCA) completed in: ", time.taken)
      # 
      # 
      # #^^^^^^^^^pca smooth with shrinkage ######################################################################
      # 
      # ############################################################################################################
      # print(Sys.time())
      # start.time <- Sys.time()
      # p.values <- numeric(ncol(sub.dat))
      # t.stats <- numeric(ncol(sub.dat))
      # threshold <- 0.05  # You can adjust this threshold
      # 
      # # Conduct voxel-wise linear regression
      # for (v in 1:ncol(sub.dat)) {
      #   model <- lm(sub.dat[, v] ~ res.var)
      #   p.values[v] <- summary(model)$coefficients[2, 4]  # Extract p-value for the slope (res.var)
      #   t.stats[v] <- abs(summary(model)$coefficients[2, 3]) #t-stats
      # }
      # 
      # t.stats[p.values > threshold] <- 0
      # 
      # var.sel <- (order(abs(t.stats), decreasing=TRUE))
      # time.taken <- Sys.time() - start.time
      # cat("MUA completed in: ", time.taken)
      
      ###raw HOLP
      print(Sys.time())
      start.time <- Sys.time()
      lambda.80 <- diag(ncol(sub.dat))
      Omeg <- solve(sub.dat%*%lambda.80%*%t(sub.dat) + (1e-4)*diag(nrow(sub.dat))) #4x4
      pre_beta_pms <- lambda.80%*%t(sub.dat)%*%Omeg #4 x 75
      beta_pms <- pre_beta_pms%*%age_tab$age
      rownames(beta_pms) <- colnames(sub.dat)
      rank_beta_pms <- beta_pms[order(abs(beta_pms), decreasing=TRUE),]
      var.sel <- (order(abs(beta_pms), decreasing=TRUE))
      time.taken <- Sys.time() - start.time
      cat("HOLP completed in: ", time.taken)
      print(Sys.time())
      
      
    }
  }
}

write.csv(results, '/well/nichols/users/qcv214/pms2/pile/sim_oct3_binmask.csv', row.names = FALSE)
