library(Rcpp)
library(RcppArmadillo)

##### Projection matrix with Shrinkage
# cppFunction('
# #define ARMA_64BIT_WORD
# 
# arma::mat proj_shrink(const arma::mat& rotations, int num_components_90, double shrinkage_param) {
# 
#   // Extract the relevant columns from the rotations matrix
#   //arma::mat selected_rotations = rotations.cols(0, num_components_90 - 1);
#   
#   // Matrix multiplication
#   //arma::mat lambda_80 = selected_rotations * selected_rotations.t();
#   
#   // Apply shrinkage to the matrix
#   //lambda_80 = shrinkage_param * lambda_80 + (1.0 - shrinkage_param) * arma::eye<arma::mat>(lambda_80.n_rows, lambda_80.n_cols);
#   arma::mat lambda_80 = shrinkage_param * (rotations.cols(0, num_components_90 - 1) * rotations.cols(0, num_components_90 - 1).t()) + (1.0 - shrinkage_param) * arma::eye<arma::mat>(rotations.n_rows, rotations.n_rows);
#   return lambda_80;
# }',depends = "RcppArmadillo")

### Efficient
cppFunction('
arma::mat proj_shrink(const arma::mat& rotations, int num_components_90, double shrinkage_param) {

  // Extract the relevant columns from the rotations matrix
  //arma::mat selected_rotations = rotations.cols(0, num_components_90 - 1);
  
  // Matrix multiplication
  //arma::mat lambda_80 = selected_rotations * selected_rotations.t();
  
  // Apply shrinkage to the matrix
  //lambda_80 = shrinkage_param * lambda_80 + (1.0 - shrinkage_param) * arma::eye<arma::mat>(lambda_80.n_rows, lambda_80.n_cols);
  arma::mat lambda_80 = shrinkage_param * (rotations.cols(0, num_components_90 - 1) * rotations.cols(0, num_components_90 - 1).t());
  
  lambda_80.diag() += (1.0 - shrinkage_param);
  
  return lambda_80;
}',depends = "RcppArmadillo")


##### Covairnace matrix with Shrinkage
cppFunction('
arma::mat cov_shrink(const arma::mat& rotations, const arma::vec& sdev, int num_components_90, double shrinkage_param) {
  
  // Extract the relevant columns from the rotations matrix
  arma::mat selected_rotations = rotations.cols(0, num_components_90 - 1);
  
  // Create the diagonal matrix with pca$sdev^2
  arma::vec sdev_squared = arma::square(sdev.subvec(0, num_components_90 - 1)); // Take square of the first num_components_90 elements of sdev
  arma::mat diag_sdev_squared = arma::diagmat(sdev_squared);  // Create the diagonal matrix
  
  // Matrix multiplication: rotations * diag(sdev^2) * t(rotations)
  arma::mat lambda_80 = selected_rotations * diag_sdev_squared * selected_rotations.t();
  
  // Apply shrinkage to the matrix
  lambda_80 = shrinkage_param * lambda_80 + (1.0 - shrinkage_param) * arma::eye<arma::mat>(lambda_80.n_rows, lambda_80.n_cols);
  
  return lambda_80;
}',depends = "RcppArmadillo")




