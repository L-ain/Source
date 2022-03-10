%% �Ӻ��� gaSVMcgForClass.m
function [BestCVaccuracy,Bestc,Bestg,ga_option] = gaSVMcgForClass(train_label,train_data,ga_option)
% gaSVMcgForClass

%
% by faruto
%Email:patrick.lee@foxmail.com QQ:516667408 http://blog.sina.com.cn/faruto BNU
%last modified 2010.01.17
%Super Moderator @ www.ilovematlab.cn

% ��ת����ע����
% faruto and liyang , LIBSVM-farutoUltimateVersion 
% a toolbox with implements for support vector machines based on libsvm, 2009. 
% Software available at http://www.ilovematlab.cn
% 
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm

% ������ʼ��
if nargin == 2
    ga_option = struct('maxgen',200,'sizepop',20,'ggap',0.9,...
        'cbound',[0,100],'gbound',[0,1000],'v',5);
end
% maxgen:���Ľ�������,Ĭ��Ϊ200,һ��ȡֵ��ΧΪ[100,500]
% sizepop:��Ⱥ�������,Ĭ��Ϊ20,һ��ȡֵ��ΧΪ[20,100]
% cbound = [cmin,cmax],����c�ı仯��Χ,Ĭ��Ϊ(0,100]
% gbound = [gmin,gmax],����g�ı仯��Χ,Ĭ��Ϊ[0,1000]
% v:SVM Cross Validation����,Ĭ��Ϊ5

%
MAXGEN = ga_option.maxgen;
NIND = ga_option.sizepop;
NVAR = 2;
PRECI = 20;
GGAP = ga_option.ggap;
trace = zeros(MAXGEN,2);

FieldID = ...
[rep([PRECI],[1,NVAR]);[ga_option.cbound(1),ga_option.gbound(1);ga_option.cbound(2),ga_option.gbound(2)]; ...
 [1,1;0,0;0,1;1,1]];

Chrom = crtbp(NIND,NVAR*PRECI);

gen = 1;
v = ga_option.v;
BestCVaccuracy = 0;
Bestc = 0;
Bestg = 0;
%
cg = bs2rv(Chrom,FieldID);

for nind = 1:NIND
    cmd = ['-v ',num2str(v),' -c ',num2str(cg(nind,1)),' -g ',num2str(cg(nind,2))];
    ObjV(nind,1) = svmtrain(train_label,train_data,cmd);
end
[BestCVaccuracy,I] = max(ObjV);
Bestc = cg(I,1);
Bestg = cg(I,2);

for gen = 1:MAXGEN
    FitnV = ranking(-ObjV);
    
    SelCh = select('sus',Chrom,FitnV,GGAP);
    SelCh = recombin('xovsp',SelCh,0.7);
    SelCh = mut(SelCh);
    
    cg = bs2rv(SelCh,FieldID);
    for nind = 1:size(SelCh,1)
        cmd = ['-v ',num2str(v),' -c ',num2str(cg(nind,1)),' -g ',num2str(cg(nind,2))];
        ObjVSel(nind,1) = svmtrain(train_label,train_data,cmd);
    end
    
    [Chrom,ObjV] = reins(Chrom,SelCh,1,1,ObjV,ObjVSel);
    
    if max(ObjV) <= 80
        continue;
    end
    
    [NewBestCVaccuracy,I] = max(ObjV);
    cg_temp = bs2rv(Chrom,FieldID);
    temp_NewBestCVaccuracy = NewBestCVaccuracy;
    
    if NewBestCVaccuracy > BestCVaccuracy
       BestCVaccuracy = NewBestCVaccuracy;
       Bestc = cg_temp(I,1);
       Bestg = cg_temp(I,2);
    end
    
    if abs( NewBestCVaccuracy-BestCVaccuracy ) <= 10^(-2) && ...
        cg_temp(I,1) < Bestc
       BestCVaccuracy = NewBestCVaccuracy;
       Bestc = cg_temp(I,1);
       Bestg = cg_temp(I,2);
    end    
    
    trace(gen,1) = max(ObjV);
    trace(gen,2) = sum(ObjV)/length(ObjV);
  
end
%
% % % figure;
% % % hold on;
% % % trace = round(trace*10000)/10000;
% % % plot(trace(1:gen,1),'r*-','LineWidth',1.5);
% % % plot(trace(1:gen,2),'o-','LineWidth',1.5);
% % % legend('�����Ӧ��','ƽ����Ӧ��',3);
% % % xlabel('��������','FontSize',12);
% % % ylabel('��Ӧ��','FontSize',12);
% % % axis([0 gen 0 100]);
% % % grid on;
% % % axis auto;
% % % 
% % % line1 = '��Ӧ������Accuracy[GAmethod]';
% % % line2 = ['(��ֹ����=', ...
% % %     num2str(gen),',��Ⱥ����pop=', ...
% % %     num2str(NIND),')'];
% % % line3 = ['Best c=',num2str(Bestc),' g=',num2str(Bestg), ...
% % %     ' CVAccuracy=',num2str(BestCVaccuracy),'%'];
% % % title({line1;line2;line3},'FontSize',12);