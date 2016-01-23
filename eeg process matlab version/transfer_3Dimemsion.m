%2维数据转为3维数据
function [transfered_data] = transfer_3Dimemsion(original_data,original_label,sample_rate,sample_time)
sample_num = size(original_label,1);
original_data((sample_rate*sample_time*sample_num+1):end,:) = [];
transfered_data = [];
for i = 1:size(original_data, 2)
    channel_normed = featureNormalize(original_data(:,i));   %归一化的方法可能要换一下
    channel_i = reshape(channel_normed, sample_rate*sample_time, sample_num);
    transfered_data(i,:,:) = channel_i;
end
end