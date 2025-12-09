MODEL COMPARISON
============================================================
               Model  Num Features R² Train R² Test RMSE Test
     sum_weight_only             1   0.0001 -0.0021   28.9057
variance_weight_only             1   0.0001 -0.0029   28.9176
    sum_and_variance             2   0.0007  0.0005   28.8681
        all_features            11   0.0104  0.0052   28.7993
============================================================

Model: sum_weight_only
Features: sum_weight
R² (train): 0.0001
R² (test): -0.0021
RMSE (test): 28.9057
Intercept: 52.5418
Coefficients:
  sum_weight: -0.000595

Model: variance_weight_only
Features: variance_weight
R² (train): 0.0001
R² (test): -0.0029
RMSE (test): 28.9176
Intercept: 51.6513
Coefficients:
  variance_weight: 0.000672

Model: sum_and_variance
Features: sum_weight,variance_weight
R² (train): 0.0007
R² (test): 0.0005
RMSE (test): 28.8681
Intercept: 53.0109
Coefficients:
  sum_weight: -0.002081
  variance_weight: 0.003491

Model: all_features
Features: sum_weight,mean_weight,median_weight,variance_weight,std_weight,min_weight,max_weight,range_weight,cv_weight,min2_weight,anchor_edge_sum
R² (train): 0.0104
R² (test): 0.0052
RMSE (test): 28.7993
Intercept: 39.6642
Coefficients:
  sum_weight: 0.000984
  mean_weight: -0.852378
  median_weight: 0.234392
  variance_weight: -0.028482
  std_weight: 1.311609
  min_weight: -0.174954
  max_weight: 0.016665
  range_weight: 0.191619
  cv_weight: 11.618933
  min2_weight: 0.255065
  anchor_edge_sum: 0.080111

