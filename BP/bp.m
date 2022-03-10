%% BP神经网络的分类
%% 清空环境变量
clear all
clc
clear
warning off

%% 导入数据

best_acc = 0;
t0 =cputime;
addpath('../Features');
Skin_features = xlsread('PCA_Skin.xlsx');
Flesh_features = xlsread('PCA_Flesh.xlsx');
%%%%Split the Skin features into HXF，MNT and MNG
skin_HXF= Skin_features(1:59,:);%%Hong-Xiangfei 59 samples；training set：41；test set：18
skin_MNT = Skin_features(60:119,:);%%Ma-Naiti 60 samples；training set：42；test set：18
skin_MNG = Skin_features(120:179,:);%%Mu-Nage 60 samples；training set：42；test set：18
%%%%Split the Flesh features into HXF，MNT and MNG
flesh_HXF = Flesh_features(1:59,:);%%Hong-Xiangfei 59 samples；training set：41；test set：18
flesh_MNT = Flesh_features(60:119,:);%%Ma-Naiti 60 samples；training set：42；test set：18
flesh_MNG = Flesh_features(120:179,:);%%Mu-Nage 60 samples；training set：42；test set：18
%%%Generate the random index to split dataset into training set and test set
a_Hxf = randperm(59); %41+18
a_Mnt = randperm(60); %42+18
a_Mng = randperm(60); %42+18

train_hxf_skin = skin_HXF(a_Hxf(1,1:41),:);%%%32+14
train_mnt_skin = skin_MNT(a_Mnt(1,1:42),:);%%%20+9
train_mng_skin = skin_MNG(a_Mng(1,1:42),:);%%14+6

train_hxf_flesh = flesh_HXF(a_Hxf(1,1:41),:);%%%32+14
train_mnt_flesh = flesh_MNT(a_Mnt(1,1:42),:);%%%20+9
train_mng_flesh = flesh_MNG(a_Mng(1,1:42),:);%%14+6


test_hxf_skin = skin_HXF(a_Hxf(1,42:end),:);
test_mnt_skin = skin_MNT(a_Mnt(1,43:end),:);
test_mng_skin = skin_MNG(a_Mng(1,43:end),:);

test_hxf_flesh = flesh_HXF(a_Hxf(1,42:end),:);
test_mnt_flesh = flesh_MNT(a_Mnt(1,43:end),:);
test_mng_flesh = flesh_MNG(a_Mng(1,43:end),:);


train_skin = [train_hxf_skin;train_mnt_skin;train_mng_skin];%%%including label
test_skin = [test_hxf_skin;test_mnt_skin;test_mng_skin];%%%including label

train_flesh= [train_hxf_flesh;train_mnt_flesh;train_mng_flesh];%%%including label
test_flesh = [test_hxf_flesh;test_mnt_flesh;test_mng_flesh];%%%including label
%%Skin and Flesh features fusion
train_dat = [train_skin(:,1:5),train_flesh(:,2:5)];
test_dat = [test_skin(:,1:5),test_flesh(:,2:5)];
%%%Obtain the training label
train_label = train_dat(:,1);
%%%Obtain to test label
test_label = test_dat(:,1);

for kk=2:size(train_dat,2)
% 训练数据
P_train = train_dat(:,2:kk)';
Tc_train = train_dat(:,1)';
T_train = ind2vec(Tc_train);

% 测试数据
P_test = test_dat(:,2:kk)';
Tc_test = test_dat(:,1)';


%% 创建网络
net = newff(minmax(P_train),[10 1],{'tansig','purelin'},'trainlm');

%% 设置网络参数
net.trainParam.epochs = 500;
net.trainParam.show = 1000;
net.trainParam.lr = 0.001;
net.trainParam.goal = 0.00001;

%% 训练网络
count_hxf = length(find(Tc_train == 1));
count_mnt = length(find(Tc_train == 2));
count_mng = length(find(Tc_train == 3));

net = train(net,P_train,Tc_train);

%% 仿真测试
T_sim = sim(net,P_test);
for i = 1:length(T_sim)
    if T_sim(i) <= 1.5
        T_sim(i) = 1;
    elseif (T_sim(i) > 1.5) && (T_sim(i) <= 2.5)
         T_sim(i) = 2;
    else
        T_sim(i) = 3;
    end
end
result = [T_sim;Tc_test];

%% 
label_1_5(kk-1,:) = T_sim;
T_sim=T_sim';Tc_test=Tc_test';


%% 计算
train_Hxf = length(find(Tc_train == 1));
train_Mnt = length(find(Tc_train == 2));
train_Mng = length(find(Tc_train == 3));
train_total = train_Hxf + train_Mnt+ train_Mng;


test_Hxf = length(find(Tc_test == 1));
test_Mnt = length(find(Tc_test == 2));
test_Mng = length(find(Tc_test == 3));
testtotal = test_Hxf  + test_Mnt + test_Mng;

all_total = train_total + testtotal ;


number_hxf = length(find(T_sim == 1 & Tc_test == 1));
number_mnt = length(find(T_sim == 2 &Tc_test == 2));
number_mng = length(find(T_sim == 3 &Tc_test == 3));

tp = length(find(T_sim ==  Tc_test));
b = tp/size(test_label,1)*100;
% b = accuracy(1,1)
if best_acc<b
    best_acc = b;
    bestkk = kk-1;
end


acc(kk-1,1) = kk-1;
acc(kk-1,2) = b
end

xlswrite('BP_pridict_result.xlsx',label_1_5);  %%all of pridict label


%%%%%Confusion matrix for feature fusion%%%%%
%%hxf
tp_hxf = length(find(T_sim(1:18,1) == 1 ));
fn_hxf = size(T_sim,1)/3 - tp_hxf;
fp_hxf = length(find(T_sim(19:end,1) == 1 ));
tn_hxf = size(T_sim(19:end,1),1)- length(find(T_sim(19:end,1) == 1 ));

hxf_precision = tp_hxf/(tp_hxf+fp_hxf)*100;
hxf_recall = tp_hxf/(tp_hxf+fn_hxf)*100;
hxf_Fscore = 2*hxf_precision*hxf_recall/(hxf_precision+hxf_recall);
%%%mnt
tp_mnt = length(find(T_sim(19:36,1) == 2 ));
fn_mnt = size(T_sim,1)/3 - tp_mnt;
fp_mnt = length(find(T_sim(1:18,1) == 2 ))+length(find(T_sim(37:54,1) == 2 ));
tn_mnt = size(T_sim(19:end,1),1)- fp_mnt;
mnt_precision = tp_mnt/(tp_mnt+fp_mnt)*100;
mnt_recall = tp_mnt/(tp_mnt+fn_mnt)*100;
mnt_Fscore = 2*mnt_precision*mnt_recall/(mnt_precision+mnt_recall);

%%%mng
tp_mng = length(find(T_sim(37:54,1) == 3 ));
fn_mng = size(T_sim,1)/3 - tp_mng;
fp_mng = length(find(T_sim(1:18,1) == 3 ))+length(find(T_sim(19:36,1) == 3));
tn_mng = size(T_sim(19:end,1),1)- fp_mng;
mng_precision = tp_mng/(tp_mng+fp_mng)*100;
mng_recall = tp_mng/(tp_mng+fn_mng)*100;
mng_Fscore = 2*mng_precision*mng_recall/(mng_precision+mng_recall);

xlswrite('BP_pridict_result.xlsx',{'HXF_fusion(%)'},2,'A7');
xlswrite('BP_pridict_result.xlsx',{'MNT_fusion(%)'},2,'A8');
xlswrite('BP_pridict_result.xlsx',{'MNG_fusion(%)'},2,'A9');
xlswrite('BP_pridict_result.xlsx',{'Precision(%)'},2,'B6');
xlswrite('BP_pridict_result.xlsx',{'Recall(%)'},2,'C6');
xlswrite('BP_pridict_result.xlsx',{'Fscore(%)'},2,'D6');
xlswrite('BP_pridict_result.xlsx',hxf_precision,2,'B7');
xlswrite('BP_pridict_result.xlsx',hxf_recall,2,'C7');
xlswrite('BP_pridict_result.xlsx',hxf_Fscore,2,'D7');

xlswrite('BP_pridict_result.xlsx',mnt_precision,2,'B8');
xlswrite('BP_pridict_result.xlsx',mnt_recall,2,'C8');
xlswrite('BP_pridict_result.xlsx',mnt_Fscore,2,'D8');

xlswrite('BP_pridict_result.xlsx',mng_precision,2,'B9');
xlswrite('BP_pridict_result.xlsx',mng_recall,2,'C9');
xlswrite('BP_pridict_result.xlsx',mng_Fscore,2,'D9');

xlswrite('BP_pridict_result.xlsx',{'Accuracy(%)'},2,'A10');
xlswrite('BP_pridict_result.xlsx',b,2,'B10');

%%%%%fusion result%%%%%
tp = length(find(T_sim ==  Tc_test));
b = tp/size(test_label,1)*100;

%%%%%Confusion matrix for skin features%%%%%
%%hxf
T_sim = label_1_5(4,:)';
tp_hxf = length(find(T_sim(1:18,1) == 1 ));
fn_hxf = size(T_sim,1)/3 - tp_hxf;
fp_hxf = length(find(T_sim(19:end,1) == 1 ));
tn_hxf = size(T_sim(19:end,1),1)- length(find(T_sim(19:end,1) == 1 ));

hxf_precision = tp_hxf/(tp_hxf+fp_hxf)*100;
hxf_recall = tp_hxf/(tp_hxf+fn_hxf)*100;
hxf_Fscore = 2*hxf_precision*hxf_recall/(hxf_precision+hxf_recall);
%%%mnt
tp_mnt = length(find(T_sim(19:36,1) == 2 ));
fn_mnt = size(T_sim,1)/3 - tp_mnt;
fp_mnt = length(find(T_sim(1:18,1) == 2 ))+length(find(T_sim(37:54,1) == 2 ));
tn_mnt = size(T_sim(19:end,1),1)- fp_mnt;
mnt_precision = tp_mnt/(tp_mnt+fp_mnt)*100;
mnt_recall = tp_mnt/(tp_mnt+fn_mnt)*100;
mnt_Fscore = 2*mnt_precision*mnt_recall/(mnt_precision+mnt_recall);

%%%mng
tp_mng = length(find(T_sim(37:54,1) == 3 ));
fn_mng = size(T_sim,1)/3 - tp_mng;
fp_mng = length(find(T_sim(1:18,1) == 3 ))+length(find(T_sim(19:36,1) == 3));
tn_mng = size(T_sim(19:end,1),1)- fp_mng;
mng_precision = tp_mng/(tp_mng+fp_mng)*100;
mng_recall = tp_mng/(tp_mng+fn_mng)*100;
mng_Fscore = 2*mng_precision*mng_recall/(mng_precision+mng_recall);

xlswrite('BP_pridict_result.xlsx',{'HXF_Skin(%)'},2,'A2');
xlswrite('BP_pridict_result.xlsx',{'MNT_Skin(%)'},2,'A3');
xlswrite('BP_pridict_result.xlsx',{'MNG_Skin(%)'},2,'A4');
xlswrite('BP_pridict_result.xlsx',{'Precision(%)'},2,'B1');
xlswrite('BP_pridict_result.xlsx',{'Recall(%)'},2,'C1');
xlswrite('BP_pridict_result.xlsx',{'Fscore(%)'},2,'D1');
xlswrite('BP_pridict_result.xlsx',hxf_precision,2,'B2');
xlswrite('BP_pridict_result.xlsx',hxf_recall,2,'C2');
xlswrite('BP_pridict_result.xlsx',hxf_Fscore,2,'D2');

xlswrite('BP_pridict_result.xlsx',mnt_precision,2,'B3');
xlswrite('BP_pridict_result.xlsx',mnt_recall,2,'C3');
xlswrite('BP_pridict_result.xlsx',mnt_Fscore,2,'D3');

xlswrite('BP_pridict_result.xlsx',mng_precision,2,'B4');
xlswrite('BP_pridict_result.xlsx',mng_recall,2,'C4');
xlswrite('BP_pridict_result.xlsx',mng_Fscore,2,'D4');
%%%%%skin result%%%%%
tp = length(find(T_sim ==  Tc_test));
b = tp/size(test_label,1)*100;

xlswrite('BP_pridict_result.xlsx',{'Accuracy(%)'},2,'A5');
xlswrite('BP_pridict_result.xlsx',b,2,'B5');

xlswrite('BP_pridict_result.xlsx',{'PCA维度'},3,'A1');
xlswrite('BP_pridict_result.xlsx',{'Accuracy(%)'},3,'B1');
xlswrite('BP_pridict_result.xlsx',acc,3,'A2');

bestkk
best_acc

