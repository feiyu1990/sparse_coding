function [X] = rand_sampling2(training, num_smp)
% sample local features for unsupervised codebook training

clabel = unique(training.label);
nclass = length(clabel);
num_img = length(training.label); % num of images
num_per_img = round(num_smp/num_img);
num_smp = num_per_img*num_img;

load(training.path{1});
dimFea = size(CNN_feature, 2);

X = zeros(dimFea, num_smp);
cnt = 0;

for ii = 1:num_img,
    fpath = training.path{ii};
    load(fpath);
    CNN_feature = permute(CNN_feature, [2, 1, 3, 4]);
    channel = size(CNN_feature, 2);
    width = size(CNN_feature, 3); height = size(CNN_feature, 4);
    feature = reshape(CNN_feature, [dimFea, width*height*channel]);
    num_fea = size(feature, 2);
    rndidx = randperm(num_fea);
    X(:, cnt+1:cnt+num_per_img) = feature(:, rndidx(1:num_per_img));
    cnt = cnt+num_per_img;
end;
