function RF_class(Path_function,traindata,testdata,trainlabel,testlabel,Feature_List)
addpath(Path_function);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%    do the RF program for 8 different channels situation:
train=traindata;    test=testdata;
for i=1:2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if i==1;	%1EEG_1通道情况
        FeatureIndex=1:38;
        mkdir('1EEG_1');
        cd('1EEG_1');
        fprintf('\t\t★★★ 1EEG_1：★★★★\n');   
    elseif i==2	%1EEG_2通道情况
        FeatureIndex=(1:38)+38;
        mkdir('1EEG_2');
        cd('1EEG_2');
        fprintf('\t\t★★★ 1EEG_2：★★★★\n');
	elseif i==3 %% 2EEG+EOG+EMG通道情况
		FeatureIndex=1:length(Feature_List);
		mkdir('2EEG+EOG+EMG');
        cd('2EEG+EOG+EMG');
        fprintf('\t\t★★★ 2EEG+EOG+EMG：★★★★\n');
	end
	%%  RF trainning and testing:
    feature_List=Feature_List(FeatureIndex); save Feature_List feature_List;
    traindata=train(:,FeatureIndex);
    testdata=test(:,FeatureIndex);
	[model,extra_options]=RFntree200(traindata,testdata,trainlabel+1,testlabel+1);
	FeatureSelect__DecreaseInAccuracy(model,traindata,testdata,trainlabel+1,testlabel+1,feature_List);
	%%	feature selected process based on SFFS
	%%%load Feature_Ordered;   
	%%%SFS(extra_options,traindata,testdata,trainlabel+1,testlabel+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd ..
end
end

function [model,extra_options]=RFntree200(traindata,testdata,trainlabel,testlabel)
%%	%%%%%%%%%%%%%%%%%%%%%%% feature selection before %%%%%%%%%%%%%%%%%%%
clear extra_options;
%extra_options.nodesize = 3;%  extra_options.nodesize = Minimum size of terminal nodes. %  Setting this number larger causes smaller trees to be grown (and thus take less time).
extra_options.predict_all = 1;
extra_options.importance = 1; %(0 = (Default) Don't, 1=calculating importance)
extra_options.localImp = 1; %(0 = (Default) Don't, 1=calculate)
extra_options.proximity = 1; %(0 = (Default) Don't, 1=calculating proximity)
extra_options.oob_prox = 0; %(Default = 1 if proximity is enabled,  Don't 0)

model = classRF_train(traindata,trainlabel,200, max(testlabel), extra_options);
%%%%
predict_label = classRF_predict(traindata,model);
accu_train=(1-length(find(predict_label~=trainlabel))/length(trainlabel))*100;
[ConfusionMatrix_train,AccuTrain_class]=getConfusionMatrix(predict_label,trainlabel);
%%%%
predict_label = classRF_predict(testdata,model);
accu_test = (1-length(find(predict_label~=testlabel))/length(testlabel))*100;
[ConfusionMatrix_test,AccuTest_class]=getConfusionMatrix(predict_label,testlabel);
%%%%
save ConfusionMatrix_BeforeSFFS extra_options predict_label accu_train ConfusionMatrix_train AccuTrain_class accu_test ConfusionMatrix_test AccuTest_class;
save('Model_BeforeFeatureSelect.mat','model','extra_options');
% 如果上面这种模型变量保存方式不可以的话，用下面这种方式保存。
% save('Model_BeforeFeatureSelect','model','extra_options','-v7.3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


function FeatureSelect__FisherScore(traindata,testdata,trainlabel,testlabel,Feature_List)
mkdir('FeatureSelect__FisherScore');    cd('FeatureSelect__FisherScore');
%%	%%%%%%%%%%%%%%%%%%%%%%%% FisherScore feature selection %%%%%%%%%%%%%%%%%%%
FeatureData=[traindata;testdata];
FeatureLabel=[trainlabel;testlabel];   %标签是从1开始的
[out] = fsFisher(FeatureData,FeatureLabel);
Importance=out.W;
save Importance_FisherScore Importance;
%%%%%%%%%%%%%%%%%
figure();
bar(out.W);xlabel('feature');ylabel('magnitude');
title('Impotrance in FisherScore');
saveas(gcf,['RFimportance_FisherScore.fig']);
close all;
%%%%%%%%%%%%%%%%    
[Importance_Order,Feature_OrderIndex]=sort(out.W ,'descend');
Feature_List_Order=Feature_List(Feature_OrderIndex);
save Feature_Ordered Feature_List_Order Feature_OrderIndex Importance_Order;
end


function FeatureSelect__DecreaseInAccuracy(model,traindata,testdata,trainlabel,testlabel,Feature_List)
mkdir('FeatureSelect__DecreaseInAccuracy');    cd('FeatureSelect__DecreaseInAccuracy');
%%	%%%%%%%%%%%%%%%%%%%%%%%% DecreaseInAccuracy feature selection %%%%%%%%%%%%%%%%%%%
[Importance_Order,Feature_OrderIndex]=sort(model.importance(:,end-1),'descend');
figure('Name','Importance Plots')
bar(model.importance(:,end-1)*100);xlabel('feature');ylabel('magnitude(%)');
title('Impotrance in DecreaseInAccuracy');
saveas(gcf,'RFimportance_DecreaseInAccuracy.fig');
close all;
%%%%%%%%%%%%%%%%%%%%%
Feature_List_Order=Feature_List(Feature_OrderIndex);
save Feature_Ordered Feature_List_Order Feature_OrderIndex Importance_Order;
cd ..
end


function FeatureSelect__DecreaseInGiniIndex(model,traindata,testdata,trainlabel,testlabel,Feature_List)
mkdir('FeatureSelect__DecreaseInGiniIndex');    cd('FeatureSelect__DecreaseInGiniIndex');
%%	%%%%%%%%%%%%%%%%%%%%%%%% DecreaseInAccuracy feature selection %%%%%%%%%%%%%%%%%%%
[Importance_Order,Feature_OrderIndex]=sort(model.importance(:,end),'descend');
figure('Name','Importance Plots')
bar(model.importance(:,end));xlabel('feature');ylabel('magnitude');
title('Impotrance in DecreaseInGiniIndex');
saveas(gcf,'RFimportance_DecreaseInGiniIndex.fig');
close all;
%%%%%%%%%%%%%%%%%%%%%
Feature_List_Order=Feature_List(Feature_OrderIndex);
save Feature_Ordered Feature_List_Order Feature_OrderIndex Importance_Order;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SFS;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cd ..
end

function SFS(extra_options,train,test,trainlabel,testlabel)
load Feature_Ordered;
FeatureSelected_Accu_class=[1 2 3 4 5 6 1 1]-1;
for Num_select=1:length(Feature_List_Order) %Num_AddOneFeature=length(select_endindex);
    Feature_Select_index = Feature_OrderIndex([1:Num_select]);
    traindata=train(:,Feature_Select_index);
    testdata =test(:,Feature_Select_index);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    model = classRF_train(traindata,trainlabel,500, max(testlabel), extra_options);
    predict_label = classRF_predict(testdata,model);
    accu_test = 1-length(find(predict_label~=testlabel))/length(testlabel);
    [ConfusionMatrix,AccuTest_class]=getConfusionMatrix(predict_label,testlabel);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    fprintf('\n前%d个顺序特征子集对应的 Accu_class：   %.2f   %.2f    %.2f    %.2f    %.2f    %.2f\n',Num_select,AccuTest_class);
    Importance=model.importance;
    eval(['save Importance_FeatureSelect_',num2str(Num_select),' Importance accu_test ConfusionMatrix AccuTest_class;']);
    FeatureSelected_Accu_class=[FeatureSelected_Accu_class;AccuTest_class,accu_test,Num_select];

end
save FeatureSelected_Accu_class FeatureSelected_Accu_class;
figure();
plot(FeatureSelected_Accu_class(2:end,7)*100);
box off;
xlabel('Number of the selected feature');ylabel('Decrease in Accuracy(%)');
title('the change with the number of selected feature');
saveas(gcf,'AccuracyChangeWithFeatureNumber.fig');
close all;
end