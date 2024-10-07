library(PMA)
library(Matrix)
# Simulate some data
X <- matrix(rnorm(1000*20000), nrow = 1000, ncol = 20000)
print("Data size")
object.size(X)
# Perform Sparse PCA using PMA
spca_res <- SPC(X, K = nrow(X)*.5, sumabsv = sqrt(ncol(X))/10 )  #if sumabsv is sqrt(p) then it's fully populated.

spca_v <- Matrix(spca_res$v, sparse = TRUE)
spca_proj <- spca_v%*%t(spca_v)

print("sparse proj size")
print(object.size(spca_proj))

pca<- prcomp(X, center = TRUE, scale. = FALSE)
pca_proj <- pca$rotation[, 1:nrow(X)]%*% t(pca$rotation[, 1:nrow(X)]) #this is 100% variance
print("normal proj size")

print(object.size(pca_proj))
