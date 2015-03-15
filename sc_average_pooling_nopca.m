function [sc_codes] = sc_average_pooling(CNN_feature, B, gamma)
%================================================
% 
% Usage:
% Compute the linear spatial pyramid feature using sparse coding. 
%
% Inputss:
% feaSet        -structure defining the feature set of an image   
%                   .feaArr     local feature array extracted from the
%                               image, column-wise
%                   .x          x locations of each local feature, 2nd
%                               dimension of the matrix
%                   .y          y locations of each local feature, 1st
%                               dimension of the matrix
%                   .width      width of the image
%                   .height     height of the image
% B             -sparse dictionary, column-wise
% gamma         -sparsity regularization parameter
% pyramid       -defines structure of pyramid 
% 
% Output:
% beta          -multiscale max pooling feature
%
% Written by Jianchao Yang @ NEC Research Lab America (Cupertino)
% Mentor: Kai Yu
% July 2008
%
% Revised May. 2010
%===============================================
dimFea = size(CNN_feature, 2);
CNN_feature = permute(CNN_feature, [2, 1, 3, 4]);
channel = size(CNN_feature, 2);
width = size(CNN_feature, 3); height = size(CNN_feature, 4);
feature = reshape(CNN_feature, [dimFea, width*height*channel]);
feature = normc(feature);
dSize = size(B, 2);
nSmp = size(feature, 2);
%img_width = feaSet.width;
%img_height = feaSet.height;
idxBin = zeros(nSmp, 1);

                 
sc_codes = zeros(dSize, nSmp);

% compute the local feature for each local feature
beta = 1e-4;
A = B'*B + 2*beta*eye(dSize);
Q = -B'*feature;

for iter1 = 1:nSmp,
    sc_codes(:, iter1) = L1QP_FeatureSign_yang(gamma, A, Q(:, iter1));
end


sc_codes = mean(sc_codes, 2);

