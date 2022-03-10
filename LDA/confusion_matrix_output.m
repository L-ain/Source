clc
clear all
close all
label_1_5 = xlsread('LDA_skin_fusion_label.xlsx');
hxf = ones(18,1);
mnt = ones(18,1)*2;
mng = ones(18,1)*3;
test_label = [hxf;mnt;mng];
label_1_5 = label_1_5';
acc = [];
for i=1:size(label_1_5,2)
    result = label_1_5(:,i);
    number_hxf = length(find(result == 0 & test_label == 0));
    number_mnt = length(find(result == 1 & test_label ==1));
    number_mng = length(find(result == 2 & test_label == 2));
    tp = length(find(result ==  test_label));
    b = tp/size(test_label,1)*100;
    acc(i,1) = i;
    acc(i,2) = b
end
xlswrite('LDA_result.xlsx',acc,2,'A1');
%%%%%fusion result%%%%%
result = label_1_5(:,8);
tp_hxf = length(find(result(1:18,1) == 1 ));
fn_hxf = size(result,1)/3 - tp_hxf;
fp_hxf = length(find(result(19:end,1) == 1 ));
tn_hxf = size(result(19:end,1),1)- length(find(result(19:end,1) == 1 ));

hxf_precision = tp_hxf/(tp_hxf+fp_hxf)*100;
hxf_recall = tp_hxf/(tp_hxf+fn_hxf)*100;
hxf_Fscore = 2*hxf_precision*hxf_recall/(hxf_precision+hxf_recall);
%%%mnt
tp_mnt = length(find(result(19:36,1) == 2 ));
fn_mnt = size(result,1)/3 - tp_mnt;
fp_mnt = length(find(result(1:18,1) == 2 ))+length(find(result(37:54,1) == 2 ));
tn_mnt = size(result(19:end,1),1)- fp_mnt;
mnt_precision = tp_mnt/(tp_mnt+fp_mnt)*100;
mnt_recall = tp_mnt/(tp_mnt+fn_mnt)*100;
mnt_Fscore = 2*mnt_precision*mnt_recall/(mnt_precision+mnt_recall);

%%%mng
tp_mng = length(find(result(37:54,1) == 3 ));
fn_mng = size(result,1)/3 - tp_mng;
fp_mng = length(find(result(1:18,1) == 3 ))+length(find(result(19:36,1) == 3));
tn_mng = size(result(19:end,1),1)- fp_mng;
mng_precision = tp_mng/(tp_mng+fp_mng)*100;
mng_recall = tp_mng/(tp_mng+fn_mng)*100;
mng_Fscore = 2*mng_precision*mng_recall/(mng_precision+mng_recall);

xlswrite('LDA_result.xlsx',{'HXF_fusion(%)'},1,'A7');
xlswrite('LDA_result.xlsx',{'MNT_fusion(%)'},1,'A8');
xlswrite('LDA_result.xlsx',{'MNG_fusion(%)'},1,'A9');
xlswrite('LDA_result.xlsx',{'Precision(%)'},1,'B6');
xlswrite('LDA_result.xlsx',{'Recall(%)'},1,'C6');
xlswrite('LDA_result.xlsx',{'Fscore(%)'},1,'D6');
xlswrite('LDA_result.xlsx',hxf_precision,1,'B7');
xlswrite('LDA_result.xlsx',hxf_recall,1,'C7');
xlswrite('LDA_result.xlsx',hxf_Fscore,1,'D7');

xlswrite('LDA_result.xlsx',mnt_precision,1,'B8');
xlswrite('LDA_result.xlsx',mnt_recall,1,'C8');
xlswrite('LDA_result.xlsx',mnt_Fscore,1,'D8');

xlswrite('LDA_result.xlsx',mng_precision,1,'B9');
xlswrite('LDA_result.xlsx',mng_recall,1,'C9');
xlswrite('LDA_result.xlsx',mng_Fscore,1,'D9');

xlswrite('LDA_result.xlsx',{'Accuracy(%)'},1,'A10');
xlswrite('LDA_result.xlsx',b,1,'B10');


tp = length(find(result ==  test_label));
b = tp/size(test_label,1)*100;
%%%%%skin result%%%%%
%%hxf
result = label_1_5(:,4);
tp_hxf = length(find(result(1:18,1) == 1 ));
fn_hxf = size(result,1)/3 - tp_hxf;
fp_hxf = length(find(result(19:end,1) == 1 ));
tn_hxf = size(result(19:end,1),1)- length(find(result(19:end,1) == 1 ));

hxf_precision = tp_hxf/(tp_hxf+fp_hxf)*100;
hxf_recall = tp_hxf/(tp_hxf+fn_hxf)*100;
hxf_Fscore = 2*hxf_precision*hxf_recall/(hxf_precision+hxf_recall);
%%%mnt
tp_mnt = length(find(result(19:36,1) == 2 ));
fn_mnt = size(result,1)/3 - tp_mnt;
fp_mnt = length(find(result(1:18,1) == 2 ))+length(find(result(37:54,1) == 2 ));
tn_mnt = size(result(19:end,1),1)- fp_mnt;
mnt_precision = tp_mnt/(tp_mnt+fp_mnt)*100;
mnt_recall = tp_mnt/(tp_mnt+fn_mnt)*100;
mnt_Fscore = 2*mnt_precision*mnt_recall/(mnt_precision+mnt_recall);

%%%mng
tp_mng = length(find(result(37:54,1) == 3 ));
fn_mng = size(result,1)/3 - tp_mng;
fp_mng = length(find(result(1:18,1) == 3 ))+length(find(result(19:36,1) == 3));
tn_mng = size(result(19:end,1),1)- fp_mng;
mng_precision = tp_mng/(tp_mng+fp_mng)*100;
mng_recall = tp_mng/(tp_mng+fn_mng)*100;
mng_Fscore = 2*mng_precision*mng_recall/(mng_precision+mng_recall);

xlswrite('LDA_result.xlsx',{'HXF_skin(%)'},1,'A2');
xlswrite('LDA_result.xlsx',{'MNT_skin(%)'},1,'A3');
xlswrite('LDA_result.xlsx',{'MNG_skin(%)'},1,'A4');
xlswrite('LDA_result.xlsx',{'Precision(%)'},1,'B1');
xlswrite('LDA_result.xlsx',{'Recall(%)'},1,'C1');
xlswrite('LDA_result.xlsx',{'Fscore(%)'},1,'D1');
xlswrite('LDA_result.xlsx',hxf_precision,1,'B2');
xlswrite('LDA_result.xlsx',hxf_recall,1,'C2');
xlswrite('LDA_result.xlsx',hxf_Fscore,1,'D2');

xlswrite('LDA_result.xlsx',mnt_precision,1,'B3');
xlswrite('LDA_result.xlsx',mnt_recall,1,'C3');
xlswrite('LDA_result.xlsx',mnt_Fscore,1,'D3');

xlswrite('LDA_result.xlsx',mng_precision,1,'B4');
xlswrite('LDA_result.xlsx',mng_recall,1,'C4');
xlswrite('LDA_result.xlsx',mng_Fscore,1,'D4');

tp = length(find(result ==  test_label));
b = tp/size(test_label,1)*100;

xlswrite('LDA_result.xlsx',{'Accuracy(%)'},1,'A5');
xlswrite('LDA_result.xlsx',b,1,'B5');




