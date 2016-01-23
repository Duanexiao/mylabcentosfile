%提取特征
feature_EEG = {'mean' 'variance' 'kurtosis' 'skewness' ...
                'spectral_d' 'spectral_rhythm4' 'spectral_rhythmRelative4' 'theta_beta' 'beta_alpha' 'thetaalpha_beta' 'thetaalpha_betaalpha' ...
                'maxV' 'zcr' 'PetrosianFD' 'HurstExp' 'PermEn' 'WaveCoef'}; 
selected_channel = [];
List_channel = cell(length(selected_channel),1);
List_head = cell(length(selected_channel),1);%需要提取特征的通道
for i = 1:length(selected_channel)
    List_channel{i,1} = feature_EEG;
    List_head{i,1} = selected_channel(i);
end
[feature_data,feature_label,feature_List] = generate_FeatureData(data,label,fs,List_channel,List_head);
save feature_data feature_data;
save feature_label feature_label;
save feature_list feature_list;

