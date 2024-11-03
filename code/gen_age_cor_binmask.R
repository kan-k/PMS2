#25 Oct, compute voxel-wise correlation against y of submask... code based of gen_HOLP


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
# install.packages("/well/nichols/users/qcv214/pms2/package/mmand_1.6.2.tar.gz", repos = NULL, type = "source")
library(mmand)


##########
##########
print("load data")
##Get pm response from kgpnn project

part_list<-read.table('/well/nichols/users/qcv214/Placement_2/participant_list.txt', header = FALSE, sep = "", dec = ".") #4529 participants
part_list$exist_vbm <- file.exists(paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',part_list[,1],'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz'))
#These two are equal
part_use<-part_list[part_list$exist_vbm==1,] #4262 participants left
part_use<-part_use[1:1000,] #only take 4k

agetab<-read.table(file = '/well/nichols/projects/UKB/SMS/ukb_latest-Age.tsv', sep = '\t', header = TRUE)
age_tab<-as.data.frame(matrix(,nrow = length(part_use$V1),ncol = 2)) #id, age, number of masked voxels
colnames(age_tab)[1:2]<-c('id','age')
age_tab$id<-part_use$V1
for(i in 1:length(part_use$V1)){
  age_tab$age[i]<-agetab$X21003.2.0[agetab$eid_8107==sub(".", "",age_tab$id[i])]
}
age_tab.test <- age_tab[501:1000,]

# age_tab <- age_tab[1:2000,]
age_tab <- age_tab[1:500,]
# age_tab.test <- age_tab[101:200,]
# age_tab <- age_tab[1:100,]


#########
#Get mask, centre of brain signal
res3.mask <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.vec <- c(fast_read_imgs_mask('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz','/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
#Defining signal to be around region 4 and 15 
mask.index <- (mask.vec %in% c(4,15))
mask.signal <- which(mask.index) #



##Get data 

list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
# sub.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
sub.dat <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))

# in.dat.mean.var <- c(mean(c(sub.dat[,mask.index])),var(c(sub.dat[,mask.index])))
# out.dat.mean.var <-  c(mean(c(sub.dat[,!mask.index])),var(c(sub.dat[,!mask.index])))

sub.dat <-sub.dat[,mask.index]

column.names <- as.character(1:ncol(sub.dat))

colnames(sub.dat) <- as.character(1:ncol(sub.dat))
list_of_all_images<-paste0('/well/win-biobank/projects/imaging/data/data3/subjectsAll/',age_tab.test$id,'/T1/T1_vbm/T1_GM_to_template_GM_mod.nii.gz')
# sub.dat.test <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
sub.dat.test <- as.matrix(fast_read_imgs_mask(list_of_all_images,'/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz'))
sub.dat.test <- sub.dat.test[,mask.index]
print("here 3")

colnames(sub.dat.test) <- colnames(sub.dat)

print("here 4")

##############################################################################################################################
######Non-smooth

# Compute voxel-wise correlation
beta_pms <- c(apply(sub.dat, 2, function(voxel_data) cor(voxel_data, age_tab$age)))

mask.temp <-oro.nifti::readNIfTI('/well/nichols/users/qcv214/pms2/sub150_centre_mask.nii.gz')
mask.temp[mask.temp %in% c(4,15)] <- beta_pms
mask.temp@datatype = 16
mask.temp@bitpix = 32
writeNIfTI(mask.temp,paste0('/well/nichols/users/qcv214/pms2/viz/oct25_age_cor_binmask_fit'))
