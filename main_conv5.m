% This is an example code for running the ScSPM algorithm described in "Linear
% Spatial Pyramid Matching using Sparse Coding for Image Classification" (CVPR'09)
%
% Written by Jianchao Yang @ IFP UIUC
% For any questions, please email to jyang29@ifp.illinois.edu.
%
% Revised May, 2010 by Jianchao Yang

clear all;
clc;

if ~exist('Results')
    mkdir('Results');
end
if ~exist('dictionary')
    mkdir('dictionary');
end
save_path = 'conv5_norelu';
train_set = 'feature_conv5_norelu_train';
test_set = 'feature_conv5_norelu_test';

%% set path
addpath('large_scale_svm');
addpath(genpath('sparse_coding'));

%% parameter setting

% directory setup
%img_dir = 'image';                  % directory for dataset images
data_dir = 'data';                  % directory to save the sift features of the chosen dataset
dataSet = train_set;
% sift descriptor extraction
skip_cal_sift = true;              % if 'skip_cal_sift' is false, set the following parameter
gridSpacing = 6;
patchSize = 16;
maxImSize = 300;
nrml_threshold = 1;                 % low contrast region normalization threshold (descriptor length)

% dictionary training for sparse coding
skip_dic_training = false;
nBases = 1024;
nsmp = 50000;
beta = 1e-5;                        % a small regularization for stablizing sparse coding
num_iters = 50;

% feature pooling parameters
pyramid = [1, 2, 4];                % spatial block number on each level of the pyramid
gamma = 0.15;
knn = 200;                          % find the k-nearest neighbors for approximate sparse coding
% if set 0, use the standard sparse coding

% classification test on the dataset
nRounds = 5;                        % number of random tests
lambda = 0.1;                       % regularization parameter for w
tr_num = 30;                        % training number per category

%rt_img_dir = fullfile(img_dir, dataSet);
rt_data_dir = fullfile(data_dir, dataSet);

%% calculate CNN fc7_conv features or retrieve the database directory

fprintf('dir the database...');
subfolders = dir(rt_data_dir);

database = [];

database.imnum = 0; % total image number of the database
database.cname = {}; % name of each class
database.label = []; % label of each class
database.path = {}; % contain the pathes for each image of each class
database.nclass = 0;

for ii = 1:length(subfolders),
subname = subfolders(ii).name;

if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
database.nclass = database.nclass + 1;

database.cname{database.nclass} = subname;

frames = dir(fullfile(rt_data_dir, subname, '*.mat'));
c_num = length(frames);

database.imnum = database.imnum + c_num;
database.label = [database.label; ones(c_num, 1)*database.nclass];

for jj = 1:c_num,
c_path = fullfile(rt_data_dir, subname, frames(jj).name);
database.path = [database.path, c_path];
end;
end;
end;
disp('done!');





disp('==================================================');
fprintf('Creating data for dictionary training...\n');
disp('==================================================');
%% load sparse coding dictionary (one dictionary trained on Caltech101 is provided: dict_Caltech101_1024.mat)
Bpath = ['dictionary/dict_' dataSet '_' num2str(nBases) '.mat'];
Xpath = ['dictionary/rand_patches_' dataSet '_' num2str(nsmp) '.mat'];

if ~skip_dic_training,
try
load(Xpath);
catch
X = rand_sampling2(database, nsmp);
save(Xpath, 'X');
end

disp('==================================================');
fprintf('Calculating the dictionary...\n');
disp('==================================================');
X = normc(X);
[B, S, stat] = reg_sparse_coding(X, nBases, eye(nBases), beta, gamma, num_iters);
save(Bpath, 'B', 'S', 'stat');
else
load(Bpath);
end

nBases = size(B, 2);                    % size of the dictionary

%% calculate the sparse coding feature

%dimFea = sum(nBases*pyramid.^2);
numFea = length(database.path);

sc_fea = zeros(nBases, numFea);
sc_label = zeros(numFea, 1);

disp('==================================================');
fprintf('Calculating the sparse coding feature...\n');
fprintf('Regularization parameter: %f\n', gamma);
disp('==================================================');



%1770 2901 4095
disp('==================================================');
fprintf('training data...\n');
disp('==================================================');
try
    load(strcat('Results/', save_path, '_sc_train_fea.mat'));
    load(strcat('Results/', save_path, '_sc_train_label.mat'));
catch
    for iter1 = 1:numFea,
        if ~mod(iter1, 50),
            fprintf('.\n');
        else
            fprintf('.');
        end;
        fpath = database.path{iter1};
        load(fpath);
        sc_fea(:, iter1) = sc_average_pooling_nopca(CNN_feature, B, gamma);
        sc_label(iter1) = database.label(iter1);
    end;
    save(strcat('Results/', save_path, '_sc_train_fea.mat'), 'sc_fea', '-v7.3');
    save(strcat('Results/', save_path, '_sc_train_label.mat'), 'sc_label', '-v7.3');
end;

train_fea = sc_fea;
train_label = sc_label;

%% evaluate the performance of the computed feature using linear SVM

[dimFea, numFea] = size(sc_fea);
clabel = unique(sc_label);
nclass = length(clabel);



%%for test%%

disp('==================================================');
fprintf('test data...\n');
disp('==================================================');

gamma = 0.15;
data_dir = 'data';                  % directory to save the sift features of the chosen dataset
dataSet = test_set;
rt_data_dir = fullfile(data_dir, dataSet);
fprintf('dir the database...');
subfolders = dir(rt_data_dir);

database = [];

database.imnum = 0; % total image number of the database
database.cname = {}; % name of each class
database.label = []; % label of each class
database.path = {}; % contain the pathes for each image of each class
database.nclass = 0;

for ii = 1:length(subfolders),
subname = subfolders(ii).name;

if ~strcmp(subname, '.') & ~strcmp(subname, '..'),
database.nclass = database.nclass + 1;

database.cname{database.nclass} = subname;

frames = dir(fullfile(rt_data_dir, subname, '*.mat'));
c_num = length(frames);

database.imnum = database.imnum + c_num;
database.label = [database.label; ones(c_num, 1)*database.nclass];

for jj = 1:c_num,
c_path = fullfile(rt_data_dir, subname, frames(jj).name);
database.path = [database.path, c_path];
end;
end;
end;
disp('done!');

numFea = length(database.path);

try
    load(strcat('Results/', save_path, '_sc_test_fea.mat'));
    load(strcat('Results/', save_path, '_sc_test_label.mat'));
catch
    for iter1 = 1:numFea,
        if ~mod(iter1, 50),
            fprintf('.\n');
        else
            fprintf('.');
        end;
        fpath = database.path{iter1};
        load(fpath);
        sc_fea(:, iter1) = sc_average_pooling_nopca(CNN_feature, B, gamma);
        sc_label(iter1) = database.label(iter1);
    end;
    save(strcat('Results/', save_path, '_sc_test_fea.mat'), 'sc_fea', '-v7.3');
    save(strcat('Results/', save_path, '_sc_test_label.mat'), 'sc_label', '-v7.3');
end;

test_fea = sc_fea;
test_label = sc_label;



%% evaluate the performance of the computed feature using linear SVM

[dimFea, numFea] = size(sc_fea);
clabel = unique(sc_label);
nclass = length(clabel);

%%%%%%%


%%%another%%%
[w, b, class_name] = li2nsvm_multiclass_lbfgs(train_fea', train_label, lambda);
[C, Y] = li2nsvm_multiclass_fwd(test_fea', w, b, class_name);
acc = zeros(length(train_label), 1);
                                
%%%%%%%%%%%%%
disp('accuracy:')
disp(sum(C==test_label)/len(test_label));

