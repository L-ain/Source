warning('off');
clear all;clc;close all;
addpath('./NIR_data/HXF');
addpath('./NIR_data/MNT');
addpath('./NIR_data/MNG');
%obtain the average NIR spectra from three kinds of rains
%%%%Hong Xiangfei Rain's NIR spectra
[Abs_HXF_Skin]=xlsread('Mean_HXF_Skin.xlsx');
[Abs_HXF_Flesh]=xlsread('Mean_HXF_Flesh.xlsx');

for i=1:size(Abs_HXF_Skin,1)
    Abs_HXF(i,:) = mean([Abs_HXF_Skin(i,:);Abs_HXF_Flesh(i,:)],1);
end
mean_hxf = mean(Abs_HXF,1);

%%%%Ma Naiti Rain's NIR spectra
[Abs_MNT_Skin]=xlsread('Mean_MNT_Skin.xlsx');
[Abs_MNT_Flesh]=xlsread('Mean_MNT_Flesh.xlsx');

for i=1:size(Abs_MNT_Skin,1)
    Abs_MNT(i,:) = mean([Abs_MNT_Skin(i,:);Abs_MNT_Flesh(i,:)],1);
end
mean_mnt = mean(Abs_MNT,1);

%%%%Mu Nage Rain's NIR spectra
[Abs_MNG_Skin]=xlsread('Mean_MNG_Skin.xlsx');
[Abs_MNG_Flesh]=xlsread('Mean_MNG_Flesh.xlsx');

for i=1:size(Abs_MNG_Skin,1)
    Abs_MNG(i,:) = mean([Abs_MNG_Skin(i,:);Abs_MNG_Flesh(i,:)],1);
end
mean_mng = mean(Abs_MNG,1);

data_all = [Abs_HXF;Abs_MNT;Abs_MNG];

label_HXF = ones(size(Abs_HXF,1),1);
label_MNT = ones(size(Abs_MNT,1),1)*2;
label_MNG = ones(size(Abs_MNG,1),1)*3;
label_all = [label_HXF;label_MNT;label_MNG];


[coff,score,latent,tq]=princomp(data_all);
m=1;
x1=latent(m);
k=length(latent);

for n=1:k
    if (x1/sum(latent))<0.99 %可控制其主成分贡献值之和，取值范围0-1，取值越接近1，提取的特征数也就越多，一般取0.8-0.9
        m=m+1;
        x1=x1+latent(m);
    else
        n=k+1;
    end
end
%% 

fprintf('  The numbers of princomp =%2d\n', m);  %%%提取的主成分数目
fprintf('  The values of princomp =%2f\n', 100*latent(1:m)/sum(latent));
fprintf('  The all values of princomp =%2f\n', 100*sum(latent(1:m))/sum(latent));
h=figure;
pareto(latent);

PCA_Dat = score(:,1:m);
data = [label_all,PCA_Dat];
xlswrite('PCA_ave_skin_flesh.xlsx',data);
