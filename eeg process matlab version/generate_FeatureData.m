function [feature_data,feature_label,feature_List]=generate_FeatureData(data,label,fs,List_channel,List_head)
%{  Note:
%---[m,N,n] = size(data);       
%---m:channel number.
%---n:epoch number.
%}

%%
    feature_data=[];
    feature_List={};
    for i=1:size(data,1)    % i:the number of channels. %%	对第m个通道上的所有数据，进行所有样本的特征提取       
        feature_channel=[];
        for j=1:size(data,3)    %j：the number of epochs
            [feature_epoch,List_now]= AMyExtract(fs,data(i,:,j),List_channel{i});
            feature_channel=[feature_channel;feature_epoch];   
        end
        feature_data = [feature_data feature_channel];
        for num=1:length(List_now)%%  Generate Feature_List for each channel
            feature_List=[feature_List;[List_head{i},'_',List_now{num}]];
        end
    end
    feature_label = label;
%%
end