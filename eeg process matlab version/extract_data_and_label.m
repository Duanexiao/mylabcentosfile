%��ȡ�Ե����ݺͱ�ǩ
clear all;clc;
hvdrfilepath = 'D:\��������\8�����ڵ�ʵ��matlab����\speech\cyq20160115';%���˴���·���ĳ�Ҫ���������·��
cd(hvdrfilepath);
vmrkfiles = dir('*.vhdr');
for m = 1:length(vmrkfiles)
    hvdrfile = ['run' num2str(m) '.vhdr'];
    display(hvdrfile)
    sampletime = 2;    %%%%�������Ҫ�Լ��ĵ�,��Ϊ���ǵ�ʵ����ȡ2s������
    eeglab;
    EEG = pop_loadbv(hvdrfilepath, hvdrfile);
    alldata = EEG.data';
    fs = EEG.srate;
    channelnum = EEG.nbchan;
    triggerposition = zeros(length(EEG.event)-2, 1);
    for i = 1:length(triggerposition)
        triggerposition(i,1) = EEG.event(i+2).latency;
    end
    %��֤ÿ������ʱ�䳤��һ��
    %%%ʵ�ʼ�¼������ÿ������ʼ������trigger֮��ļ�������ϸ����2s��
    eval(['data' num2str(m) '= [];']);
    for j = 1:2:length(triggerposition)
        eval(['data' num2str(m) '=[' 'data' num2str(m) ';' 'alldata(triggerposition(j, 1)+1:triggerposition(j, 1) + fs * sampletime,:)];']);
    end
    eval(['save data' num2str(m) ' data' num2str(m) ';']);
    samplenum = eval(['size(data' num2str(m) ',1)'])/(fs*sampletime);
    eval(['label' num2str(m) '=zeros(samplenum, 1);']);
    for k = 1:samplenum
        if  eval(EEG.event(2*k+1).type(4:end)) == 1
            eval(['label' num2str(m) '(k, 1) = 1;']);
        elseif eval(EEG.event(2*k+1).type(4:end)) == 3
            eval(['label' num2str(m) '(k, 1) = 2;']);
        elseif eval(EEG.event(2*k+1).type(4:end)) == 5
            eval(['label' num2str(m) '(k, 1) = 3;']);
        elseif eval(EEG.event(2*k+1).type(4:end)) == 7
            eval(['label' num2str(m) '(k, 1) = 4;']);
        elseif eval(EEG.event(2*k+1).type(4:end)) == 9
            eval(['label' num2str(m) '(k, 1) = 5;']);
        elseif eval(EEG.event(2*k+1).type(4:end)) == 11
            eval(['label' num2str(m) '(k, 1) = 6;']);
        elseif eval(EEG.event(2*k+1).type(4:end)) == 13
            eval(['label' num2str(m) '(k, 1) = 7;']);   
       elseif eval(EEG.event(2*k+1).type(4:end)) == 15
            eval(['label' num2str(m) '(k, 1) = 8;']);     
        end
    end
    eval(['save label' num2str(m) ' label' num2str(m) ';']);
    close('EEGLAB v12.0.2.6b')  %�ر�EEGLAB��ע��EEGLAB�汾�����⣬Ҫ�ĳ���Ӧ�汾�����򱨴�
end
%datafile = dir('data*.mat');
%finaldata = zeros(size(data1, 2), size(data1, 1), length(datafile)*samplenum);  %ͨ����*��������*������

%for i = 1: length(datafile)
    %eval(['finaldata = [finaldata;' 'data' num2str(i) '];']);
    %finaldata(:,:,(i-1)*samplenum+1:i*samplenum) = eval(['data' num2str(i)])';
%end

labelfile = dir('label*.mat');
finallabel = [];
finaldata = [];
for i = 1: length(labelfile)
    eval(['finaldata = [finaldata;' 'data' num2str(i) '];']);
    eval(['finallabel = [finallabel;' 'label' num2str(i) '];']);
end
save finaldata finaldata;
save finallabel finallabel;