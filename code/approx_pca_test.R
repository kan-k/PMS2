x.test <- matrix(rnorm(3*5),nrow=5)
cor(x.test)
cov(x.test)
x.pca <- prcomp(x.test)
x.pca$rotation%*% t(x.pca$rotation) #this thing gives an approx of identity matrix!, bot cor or cov. This is identity since full proj means using raw data

x.pca$rotation %*% diag(x.pca$sdev^2) %*% t(x.pca$rotation) #gives the exact cov matrix


x.pca$rotation[,1:2] %*% diag(x.pca$sdev[1:2]^2) %*% t(x.pca$rotation[,1:2])


x.pca2 <- prcomp(x.test, scale. = TRUE)
x.pca2$rotation %*% diag(x.pca2$sdev^2) %*% t(x.pca2$rotation)
