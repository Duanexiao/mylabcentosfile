function [train_data, train_label, test_data, test_label] = train_test_split(pre_data, pre_label, train_size)
    label_num = length(unique(pre_label));
    diff_label = unique(pre_label);
    train_data = [];
    train_label = [];
    test_data = [];
    test_label = [];
    for i = 1:label_num
        eval(['index', num2str(i), '=find(pre_label==', num2str(diff_label(i)) ');']);
        eval(['shuffled_index', num2str(i), '=shuffle(index', num2str(i), ');']);
        eval(['train_num', '=floor(train_size*length(shuffled_index' num2str(i) '));']);    %length是获取行列中较大的值！！！
        eval(['train_data', num2str(i), '=pre_data(shuffled_index', num2str(i), '(1:train_num),:);']);
        eval(['train_label', num2str(i), '=pre_label(shuffled_index', num2str(i), '(1:train_num),:);']);
        eval(['test_data', num2str(i), '=pre_data(shuffled_index', num2str(i), '(train_num+1:end),:);']);
        eval(['test_label', num2str(i), '=pre_label(shuffled_index', num2str(i), '(train_num+1:end),:);']);
        eval(['train_data=', '[train_data;train_data', num2str(i), '];']);
        eval(['train_label=', '[train_label;train_label', num2str(i), '];']);
        eval(['test_data=', '[test_data;test_data', num2str(i), '];']);
        eval(['test_label=', '[test_label;test_label', num2str(i), '];']);
    end
    train_num = size(train_data,1);
    shuffled_index = randperm(train_num);
    train_data = train_data(shuffled_index,:);
    train_label = train_label(shuffled_index,:);
end
