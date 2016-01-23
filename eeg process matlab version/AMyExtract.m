function [feature,List]= AMyExtract(SampleRate,data,mode)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
feature_names = mode;
feature=[];
List={};
for i=1:length(feature_names)
    mode=feature_names{i};
    switch mode
        case 'mean'
            tmpE = mean(data);
            tmpList = mode;
        case 'std'
            tmpE = std(data);
            tmpList = mode;
        case 'kurtosis'
            tmpE = kurtosis(data);
            tmpList = mode;
        case 'skewness'
            tmpE = skewness(data);
            tmpList = mode;
        case 'amp'  %% 幅值
            tmpE = abs(max(data)-min(data));
            tmpList = mode;
        case 'absmedian'
            tmpE = median(abs(data));
            tmpList = mode;
        case 'variance'
            tmpE = var(data);
            tmpList = mode;
%         case 'zcr'  %% 过零率，在下面有
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'relativePower5'
            [f Y] = fft2psd(data,SampleRate);  %得到power spectral density（psd）功率谱密度：单位频率上的功率Y
            nP1 = Y(f>=0.5 & f<4);  % delta
            nP2 = Y(f>=4 & f<8);    % theta
            nP3 = Y(f>=8 & f<13);   % alpha
            nP4 = Y(f>=13 & f<20);  % beta
            nP5 = Y(f>=20 & f<SampleRate/2); % high freq    %%注意：这里的SampleRate/2=fn，即为奈奎斯特频率
            nP = sum( Y(f>=0.5 & f<SampleRate/2) );
            tmpE = [sum(nP1) sum(nP2) sum(nP3) sum(nP4) sum(nP5)]/nP;
            
        case 'spectral_d'   %%指定频带范围内的功率谱能量
            [f Y] = fft2psd(data,SampleRate);
            tmpE = sum( Y(f>=0.01 & f<30) );
            tmpList = mode;
        case 'spectral_rhythm4'
            [f Y] = fft2psd(data,SampleRate);  % 得到power spectral density（psd）功率谱密度：单位频率上的功率Y
            nP1 = Y(f>=0.01 & f<4);  % delta
            nP2 = Y(f>=4 & f<8);    % theta
            nP3 = Y(f>=8 & f<12);   % alpha
            nP4 = Y(f>=12 & f<30);  % beta
            tmpE =[sum(nP1) sum(nP2) sum(nP3) sum(nP4)];    %nP = sum( Y(f>=0.01 & f<30) );
            tmpList = {[mode,'_delta'];[mode,'_theta'];[mode,'_alpha'];[mode,'_beta']};
        case 'spectral_rhythmRelative4'
            [f Y] = fft2psd(data,SampleRate);  % 得到power spectral density（psd）功率谱密度：单位频率上的功率Y
            nP1 = Y(f>=0.01 & f<4);  % delta
            nP2 = Y(f>=4 & f<8);    % theta
            nP3 = Y(f>=8 & f<12);   % alpha
            nP4 = Y(f>=12 & f<30);  % beta
            nP = sum( Y(f>=0.01 & f<30) );
            tmpE =[sum(nP1) sum(nP2) sum(nP3) sum(nP4)]/nP;  
            tmpList = {[mode,'_delta'];[mode,'_theta'];[mode,'_alpha'];[mode,'_beta']};
        case 'relativePower10'
            [f Y] = fft2psd(data,SampleRate);  %得到power spectral density（psd）功率谱密度：单位频率上的功率Y
            nP1=Y(f>=0.5 & f<2);    % delta1
            nP2=Y(f>=2 & f<4);      % delta2
            nP3=Y(f>=4 & f<6);      % theta1
            nP4=Y(f>=6 & f<8);      % theta2
            nP5=Y(f>=8 & f<10);     % alpha1
            nP6=Y(f>=10 & f<12);    % alpha2
            nP7=Y(f>=12 & f<14);    % sigma1
            nP8=Y(f>=14 & f<16);    % sigma2
            nP9=Y(f>=16 & f<30);    % beta
            nP10=Y(f>=30 & f<SampleRate/2);  % gamma    %%注意：这里的SampleRate/2=fn，即为奈奎斯特频率
            nP = sum( Y(f>=0.5 & f<SampleRate/2) );
            tmpE = [sum(nP1) sum(nP2) sum(nP3) sum(nP4) sum(nP5) sum(nP6) sum(nP7) sum(nP8) sum(nP9) sum(nP10)]/nP;
%             tmpList = {[mode,'_delta'];[mode,'_theta'];[mode,'_alpha'];[mode,'_beta']};

        case 'theta_beta'
            [f Y] = fft2psd(data,SampleRate);
            nP2 = Y(f>=4 & f<8);    % theta
            nP4 = Y(f>=12 & f<30);  % beta
            tmpE = sum(nP2)/sum(nP4);
            tmpList = mode;
        case 'beta_alpha'
            [f Y] = fft2psd(data,SampleRate); 
            nP4 = Y(f>=12 & f<30);  % beta
            nP3 = Y(f>=8 & f<12);   % alpha
            tmpE = sum(nP4)/sum(nP3);
            tmpList = mode;
        case 'thetaalpha_beta'
            [f Y] = fft2psd(data,SampleRate); 
            nP2 = Y(f>=4 & f<8);    % theta
            nP3 = Y(f>=8 & f<12);   % alpha
            nP4 = Y(f>=12 & f<30);  % beta
            tmpE = (sum(nP2)+sum(nP3))/sum(nP4);
            tmpList = mode;
        case 'thetaalpha_betaalpha'
            [f Y] = fft2psd(data,SampleRate);
            nP2 = Y(f>=4 & f<8);    % theta
            nP3 = Y(f>=8 & f<12);   % alpha
            nP4 = Y(f>=12 & f<30);  % beta
            tmpE = (sum(nP2)+sum(nP3))/(sum(nP4)+sum(nP3));
            tmpList = mode;

        case 'sweat2'  %返回的两个特征值的含义分别对应：功率谱中的最大值、该最大值对应的频率位置
            [f Y] = fft2psd(data,SampleRate);
            [maxY loc] = max(Y);
            tmpE = [maxY f(loc)];
%             tmpList = mode;


        case 'se95fre'  
            % 在经过[1,SampleRate/2]的滤波之后，在从功率谱层面上，统计累计能量第一次达到整个带宽上（功率谱）能量95%以上的频率位置。注意：重点是累计能量
            % 然后将这一统计得到的频率值作为当前要统计的特征
            dataBP=BPfilter(double(data),[1 SampleRate/2],SampleRate);  %% 注意：这里的数据b必须是double类型的数据，否则会报错
            [f Y] = fft2psd(data,SampleRate);   % 先得到功率谱密度图：plot(f,Y)
            Yhalf=Y(1:end/2);   fhalf=f(1:end/2);   N=length(Yhalf);
            Etot=sum(Yhalf.^2); % 计算整个可统计的频带范围上的功率谱总能量
            psi=zeros(N,1); %存放 累计能量的相对比值
            for n=1:N
                psi(n)=sum(Yhalf(1:n).^2)/Etot;
            end
            temp=find(psi>0.95);
            tmpE=fhalf(temp(1)); 
            tmpList = mode;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'entropy'  
            %%% 法一：
            N = length(data);
            P = 1/N * hist(data,ceil(sqrt(N)));   % hist(,)这个函数的主要作用：将数据按照其值的范围均等的分成ceil(sqrt(N))份，返回每一均等范围内的数据个数。
            tmpE = -nansum( P.*log(P) );
            tmpList = mode;
    %         %%% 法二：
    %         tmpE = wentropy(b,'shannon');  % 计算香农熵


        case 'dwt4'
            w = 'sym3';     % Wavelet mother function
            l = 1;          % DWT-level
            [C, L] = wavedec(b,l,w);
            A1=C(1:L(1));
            %D1=C(L(1)+1:L(3));
            tmpE =[max(A1) min(A1) mean(A1) std(A1)]; % max(D1) min(D1) mean(D1) std(D1)];
%             tmpList = mode;

            %%%%%% 可继续补充，把自己以前写的提取小波系数的程序补充到这里
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        case 'fractalexp'   %%一种分形指数的线性估计方法：用当前数据，对其谱密度进行一元一次线性拟合，用拟合中的负斜率，来近似估计当前数据的分形指数
            [f Y] = fft2psd(data,SampleRate);
            f=f(f>=0.5 & f<SampleRate/2);
            Y=Y(f>=0.5 & f<SampleRate/2);
            P=polyfit(log(f),log(Y),1);    %进行一元一次线性拟合，拟合结果P中，第一个数值是斜率，第二个数值是截距
            tmpE = P(1);
            tmpList = mode;
%%%%%%%%%%%%%%%%%%%%%%★20140815：补充的特征★%%%%%%%%%%%%%%%%%%%%%%%
        case 'zcr'
            tmpE = compute_zrc(data);tmpList = mode;
        case 'maxV'
            tmpE = max(data);tmpList = mode;
        case 'PetrosianFD'     %%计算一种分形维数，叫PetrosianFDimension，简称为PetrosianFD
            tmpE = compute_PetrosianFractalDimension(data);tmpList = mode;
        case 'HurstExp'     %%计算HurstExponent指数
            tmpE = compute_HurstExponent(data);tmpList = mode;
        case 'PermEn'   %%计算排列熵，这里的参数order就是需要进行排列的阶数，对应有order！这么多种排列组合，所以order的值不能太大
            order=3;
            t=SampleRate*1; %%一般t选择的数据长度，一般选择一秒时间长度的延迟即可
%             [tmpE hist] = compute_PermutationEntropy(data,order,t);
            tmpE = compute_PermutationEntropy(data,order,t);tmpList = mode;
        case 'WaveCoef'
            if SampleRate==128
                absmean=mean(abs(data));
                [C,L]=wavedec(data,5,'db4');   %先直接进行多层小波分解,这里，因为采样率为128Hz,所以进行了5层小波分解
% % %                 [d1,d2,d3,d4,d5]=detcoef(C,L,[1 2 3 4 5]); %cd1~cd5分别对应的是data数据进行小波分解后的第1~第5层的高频部分
                d3=detcoef(C,L,3);    % beta:16~32
                d4=detcoef(C,L,4);    % alpha：8~16
                d5=detcoef(C,L,5);    % theta :4~8Hz
                a5=appcoef(C,L,'db4',5); 
                D3=[mean(d3),sum(d3.^2),std(d3),sum(abs(d3))/absmean];
                D4=[mean(d4),sum(d4.^2),std(d4),sum(abs(d4))/absmean];
                D5=[mean(d5),sum(d5.^2),std(d5),sum(abs(d5))/absmean];
                A5=[mean(a5),sum(a5.^2),std(a5),sum(abs(a5))/absmean];
                tmpE =[D3 D4 D5 A5];           
                tmpList = {[mode,'_delta','_1'];[mode,'_delta','_2'];[mode,'_delta','_3'];[mode,'_delta','_4'];... 
                    [mode,'_theta','_1'];[mode,'_theta','_2'];[mode,'_theta','_3'];[mode,'_theta','_4'];... 
                    [mode,'_alpha','_1'];[mode,'_alpha','_2'];[mode,'_alpha','_3'];[mode,'_alpha','_4'];...
                    [mode,'_beta','_1'];[mode,'_beta','_2'];[mode,'_beta','_3'];[mode,'_beta','_4'];};
            elseif SampleRate==500
                absmean=mean(abs(data));
                [C,L]=wavedec(data,6,'db4');   %采样频率为500Hz的，最多我们只能观察到250Hz的信号频率范围，所以，要进行6次小波分解
% % %                 [d1,d2,d3,d4,d5,d6]=detcoef(C,L,[1 2 3 4 5 6]); %cd1~cd5分别对应的是data数据进行小波分解后的第1~第5层的高频部分
                d4=detcoef(C,L,4);    % beta:16~32 → 15.6~31.25Hz
                d5=detcoef(C,L,5);    % alpha：8~16 → 7.8~15.6Hz
                d6=detcoef(C,L,6);    % theta :4~8Hz → 3.9~7.8Hz
                a6=appcoef(C,L,'db4',6); % delta:0~4Hz → 0~3.9Hz
                D4=[mean(d4),sum(d4.^2),std(d4),sum(abs(d4))/absmean];
                D5=[mean(d5),sum(d5.^2),std(d5),sum(abs(d5))/absmean];
                D6=[mean(d6),sum(d6.^2),std(d6),sum(abs(d6))/absmean];
                A6=[mean(a6),sum(a6.^2),std(a6),sum(abs(a6))/absmean];
                tmpE =[D4 D5 D6 A6];
                tmpList = {[mode,'_delta','_1'];[mode,'_delta','_2'];[mode,'_delta','_3'];[mode,'_delta','_4'];... 
                    [mode,'_theta','_1'];[mode,'_theta','_2'];[mode,'_theta','_3'];[mode,'_theta','_4'];... 
                    [mode,'_alpha','_1'];[mode,'_alpha','_2'];[mode,'_alpha','_3'];[mode,'_alpha','_4'];...
                    [mode,'_beta','_1'];[mode,'_beta','_2'];[mode,'_beta','_3'];[mode,'_beta','_4'];};
            end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        case 'corr'
            tmpE = xcorr(data(1,:),data(2,:),0,'coef');
            tmpList = mode;
%%%%%%%%%%%%%%%%%%%%%%★20140825：补充的特征★%%%%%%%%%%%%%%%%%%%%%%%
        case 'HA'
            tmpE = var(data);tmpList = mode;
        case 'HM'
            tmpE = std(diff(data))/std(data);tmpList = mode;
        case 'HC'
            tmpE = abs(std(diff(data,2))/std(diff(data)))-(std(diff(data))-std(data)).^2;tmpList = mode;

        case 'MeanTimeEnergy'
            MTimeE=sum(data.^2)/length(data);
            tmpE = MTimeE;tmpList = mode;
        case 'MeanTeagerEnergy'
            N=length(data);
            sumV=0;
            for k=3:N
                sumV = data(i-1).^2-data(i)*data(i-2)+sumV;
            end
            MTeagerE=sumV/N;
            tmpE = MTeagerE;tmpList = mode;
        case 'MeanCurveLength'
            N=length(data);
            sumV=0;
            for k=2:N
                sumV=abs(data(i)-data(i-1))+sumV;
            end
            MCL=sumV/N;
            tmpE = MCL;tmpList = mode;
%         case 'SpectralEn'
%         case 'ReliEn'
        case 'ApEn'
%             tmpE=compute_ApEn(date);
            tmpE = compute_ApEn_fast(data);tmpList = mode;
        case 'SaEn'
            tmpE=compute_SaEn(date);tmpList = mode;
    end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
feature = [feature tmpE];
List = [List;tmpList];
end

end


%%
function [f psd] = fft2psd(data,SampleRate)

    N   = length(data);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    f   = SampleRate*(0:N-1)/N;
    psd = abs(fft(data)).^2 /N /SampleRate;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function afterBPfilter = BPfilter(data,freq,SampleRate) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BPfilter：2nd-order bandpass filter. 进行二阶BP滤波
% freq：是要进行BP滤波的频率范围，应该是个一行两列的数组
% SampleRate：数据的实际采样率  fn是奈奎斯特频率，fn数值上=SampleRate/2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    filtorder = 2;
    if freq(1) > 0 && freq(2) >= SampleRate/2
        [b, a] = butter(filtorder, freq(1)*(2/SampleRate), 'high');
    elseif freq(1) <= 0 && freq(2) < SampleRate/2
        [b, a] = butter(filtorder, freq(2)*(2/SampleRate), 'low');
    elseif freq(1) > 0 && freq(2) < SampleRate/2
        [b, a] = butter(filtorder, freq*(2/SampleRate));
    else
        disp('ERROR: Cut-off frequency');
        return
    end
    %disp(['Filtered successfully with a: ' num2str(a) ' b: ' num2str(b)]);
    afterBPfilter = filtfilt(b, a, data);

end

function data_zcro = compute_zrc(data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 当data只是个一维向量时：
    data_zcro=0;
    if data(1)==0
        data_zcro=data_zcro+1;
    end
    for j=2:length(data);
        if data(j)*data(j-1)<0 || data(j)==0;
            data_zcro=data_zcro+1;
        end
    end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%% 当data只是个二维向量时，有i行，j列：可用以下计算代码计算过零率
% %%% 该过零率的计算，是对每一行的数据都算出该行数据中过零点的个数，即为该行数据的过零率
% data_zcro=zeros(size(data,1),1);
% for i=1:size(data,1)
%     z=data(i,:);    %%z是对应的一维向量
%     %以下是对一个一维向量计算其过零率
%     if z(1)==0
%         data_zcro(i)=data_zcro(i)+1;
%     end
%     for j=2:length(z);
%         if z(j)*z(j-1)<0 || z(j)==0;
%             data_zcro(i)=data_zcro(i)+1;
%         end
%     end
% end
    
end

function PetrosianFD = compute_PetrosianFractalDimension(data)
%%  理论基础，详见  文献【40】：Kolmogorov Complexity of Finite Sequences and Recognition of Different Preictal EEG Patterns 
%%%%%%%%%%%%%%%%%%   petrosian fractal dimension，相当于是对一般的有限长序列计算其Kolmogorov Complexity复杂度的一种快速算法
%%  首先：
%%%  先对data数据进行二值化处理，
%%%  这里用的二进制化的处理方法是：大于均值的数据，赋值为1；小于均值的数据，赋值为0.
average=mean(data);
data(data>average)=1;   % data(data<=average)=0;
%% 用计算过零率的方法来计算Nδ：
%   这里的N_delta的含义是发生了0与1之间的跳变，这里为了方便用计算过零率的方法来计算跳变个数，可以先将本来为0的值赋值为-1.
data(data<=average)=-1;
N_delta = compute_zrc( data );
%%  用petrosian提出的方法（下面的公式）进行对以有限长序列的Kolmogorov Complexity复杂度计算，即得到PFD（petrosian fractal dimension）
k = length(data);
PetrosianFD = log(k)/(log(k)+log(k/(k+0.4*N_delta)));
PetrosianFD = PetrosianFD/log(10);

end

function HurstExp = compute_HurstExponent(data)
% [M,npoints]=size(data); %%其实只用到了npoints这个参数
npoints=length(data);
yvals=zeros(1,npoints); 
xvals=zeros(1,npoints); 
data_next=zeros(1,npoints); 
 
index=0; 
binsize=1; 
while npoints>5 	%%这是平均延迟3的情况，所以，当npoints
    index=index+1; 
    xvals(index)=binsize;     
    yvals(index)=binsize*std(data);

    npoints=fix(npoints/2);     
    for ipoints=1:npoints % average adjacent points in pairs 
        data_next(ipoints)=(data(2*ipoints)+data((2*ipoints)-1))*0.5; 
    end
    data = data_next(1:npoints); 
    binsize=binsize*2; 
     
end % while 

xvals=xvals(1:index);   yvals=yvals(1:index); 
logx=log(xvals);        logy=log(yvals); 
p2=polyfit(logx,logy,1); 
HurstExp = p2(1); % Hurst exponent is the slope of the linear fit of log-log plot 
return;

end

function PermEn = compute_PermutationEntropy(data,order,t)

N = length(data);
permlist = perms(1:order);  %生成一个【m！* m】维的排列组合顺序矩阵
c(1:length(permlist))=0; 
 for j=1 : N-t*(order-1)
     [a,iv]=sort(data(j:t:j+t*(order-1)));
     for jj=1:length(permlist)
         if (abs(permlist(jj,:)-iv))==0
             c(jj) = c(jj) + 1 ;
         end
     end
 end
hist = c;
c=c(find(c~=0));
p = c/sum(c);
PermEn = -sum(p .* log(p));

end

function [ApEn_value] = compute_ApEn_fast(signal,r_factor)
% function [ApEn_value,Cmr,Cmr_1] = fast_ApEn(signal,0.2)
% Estimate the Aproximate Entropy (ApEn) of a signal, using a fast algorithm, for the ApEn parameter "m" equal to 2 
% The pattern length "m" for which this routine was implemented is 2. For another values of "m", the instructions marked with a (*) in the end must be changed.			
% m=1 or m=2
% r between 0.1*STD and 0.25*STD, where STD is the signal standard deviation 
% N (signal length) between 75 and 5000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if length(signal)<75 | length(signal)>5000
% slength=input('Signal length inappropriate. Continue anyway? (y/n)','s');
% if strcmpi(slength,'y')==0
% return
% end
% end

% if r_factor<0.1 | r_factor>0.25
%     r_factor_flag=input('Value for r parameter is inappropriate. Continue anyway? (y/n)','s');
%     if strcmpi(r_factor_flag,'y')==0
%         return
%     end
% end

% Initial variables definition.
if nargin==1
    r_factor=0.2;
end
signal=signal(:);
N=length(signal);
Cmr_ij=[];Cmr_i=[];Cmr=[];
Cmr_ij_1=[];Cmr_i_1=[];Cmr_1=[];
% D and S matrixes computation.
D=abs(signal*ones(1,N)-ones(N,1)*signal');
S=zeros(N,N);S(find(D<=r_factor*std(signal)))=1;
% C's computation for "m" and "m+1" patterns.
m=2;
S(N+1,(m+1):N)=0; % necessary for the loop to be possible ("artificial" definitions)
for k=1:N-(m-1)% m pattern.
    Cmr_ij=S(k,1:N-1).*S(k+1,2:N); % (*)
    Nm_i=sum(Cmr_ij);
    Cmr_i=Nm_i/(N-(m-1));
    Cmr=[Cmr; Cmr_i];% m+1 pattern.
    Cmr_ij(end)=[];
    Cmr_ij_1=Cmr_ij.*S(k+2,3:N);
    Nm_i_1=sum(Cmr_ij_1);
    Cmr_i_1=Nm_i_1/(N-m);
    Cmr_1=[Cmr_1;Cmr_i_1];
end
Cmr_1(end)=[]; % the last C value for the "m+1" pattern is artificial.% Phi’s computation.
phi_m=mean(log(Cmr));
phi_m_1=mean(log(Cmr_1));% ApEn final calculation.
ApEn_value=[phi_m-phi_m_1];% phi_m% phi_m_1
end

function [shang]=compute_SaEn(xdate)
m=2;
n=length(xdate);
r=0.2*std(xdate);
cr=[];
gn=1;
gnmax=m;
while gn<=gnmax
    x2m=zeros(n-m+1,m);%存放变换后的向量
    d=zeros(n-m+1,n-m);% 存放距离结果的矩阵
    cr1=zeros(1,n-m+1);%存放
    k=1;
    for i=1:n-m+1
        for j=1:m
            x2m(i,j)=xdate(i+j-1);
        end
    end
    for i=1:n-m+1
        for j=1:n-m+1
            if i~=j
                d(i,k)=max(abs(x2m(i)-x2m(j))); %计算各个元素和响应元素的距离
                k=k+1;
            end
        end
        k=1;
    end
    for i=1:n-m+1
        [k,l]=size(find(d(i<r)));%将比R小的个数传送给L
        cr1(1,i)=l;
    end
    cr1=(1/(n-m))*cr1;
    sum1=0;
    for i=1:n-m+1
        if cr1(i)~=0
            sum1=sum1+log(cr1(i));
        end
    end
    cr1=1/(n-m+1)*sum1;
    cr(1,gn)=cr1;
    gn=gn+1;
    m=m+1;
end
% shang=cr(1,1)-cr(1,2);
shang=-log(cr(1,1)/cr(1,2));
end



