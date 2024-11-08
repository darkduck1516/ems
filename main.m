warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行
tic
% restoredefaultpath
%%导入数据
 data = xlsread('负载2号位实验.xlsx');
 load('dataN.mat');%matrix
 YSJ= dataN;
 
  %% 数据预处理，数据可能是存储在矩阵或者是EXCEL中的二维数据，衔接为一维的，如果数据是一维数据，此步骤也不会影响数据
 
 [c,l]=size(YSJ);
 
 Y=[];
 
 for i=1:c
 
     Y=[Y,YSJ(i,:)];
 
 end
 
 [c1,l1]=size(Y);
 
 X=[1:l1];
 
 

 %% 绘制噪声信号图像

 
  figure(1);
  
  plot(X,Y);
  
  xlabel('横坐标');
  
  ylabel('纵坐标');
  
  title('原始信号');
 
  %% 硬阈值处理
 
 lev=3;
 
 xd=wden(Y,'heursure','h','one',lev,'db4');%硬阈值去噪处理后的信号序列
 
  figure(2)
  
  plot(X,xd)
  
  xlabel('横坐标');
  
  ylabel('纵坐标');
  
  title('硬阈值去噪处理')
  
  set(gcf,'Color',[1 1 1])
 
  %% 软阈值处理
 
 lev=3;
 
 xs=wden(Y,'heursure','s','one',lev,'db4');%软阈值去噪处理后的信号序列
 
 figure(3)
 
 plot(X,xs)
 
 xlabel('横坐标');
 
 ylabel('纵坐标');
 
 title('软阈值去噪处理')
 
 set(gcf,'Color',[1 1 1])

 %% 固定阈值后的去噪处理
 
 lev=3;
 
 xz=wden(Y,'sqtwolog','s','sln',lev,'db4');%固定阈值去噪处理后的信号序列
 
 % figure(4)
 % 
 % plot(X,xz);
 % 
 % xlabel('横坐标');
 % 
 % ylabel('纵坐标');
 % 
 % title('固定阈值后的去噪处理')
 % 
 % set(gcf,'Color',[1 1 1])
 
 %% 计算信噪比SNR
 
 Psig=sum(Y*Y')/l1;
 
 Pnoi1=sum((Y-xd)*(Y-xd)')/l1;
 
 Pnoi2=sum((Y-xs)*(Y-xs)')/l1;
 
 Pnoi3=sum((Y-xz)*(Y-xz)')/l1;
 
 SNR1=10*log10(Psig/Pnoi1);
 
 SNR2=10*log10(Psig/Pnoi2);
 
 SNR3=10*log10(Psig/Pnoi3);
 
 %% 计算均方根误差RMSE
 
 RMSE1=sqrt(Pnoi1);
 
 RMSE2=sqrt(Pnoi2);
 
 RMSE3=sqrt(Pnoi3);
 
 %% 输出结果
 
 disp('-------------三种阈值设定方式的降噪处理结果---------------'); 
 
 disp(['硬阈值去噪处理的SNR=',num2str(SNR1),'，RMSE=',num2str(RMSE1)]);
 
 disp(['软阈值去噪处理的SNR=',num2str(SNR2),'，RMSE=',num2str(RMSE2)]);
 
 disp(['固定阈值后的去噪处理SNR=',num2str(SNR3),'，RMSE=',num2str(RMSE3)]);


%%导入数据




 num_samples = length(xs);       % 样本个数 
 kim = 15;                       % 延时步长（kim个历史数据作为自变量）
 zim =  1;                      % 跨zim个时间点进行预测
 or_dim = size(xs,2);            %求矩阵X列数的大小
 

 for i = 1: num_samples - kim - zim + 1   
     res(i, :) = [reshape(data(i: i + kim - 1), 1,kim), data(i + kim + zim - 1)];
 end



%%  数据分析
num_size = 0.875;                              % 训练集占数据集比例
outdim = 1;                                  % 最后一列为输出
num_samples = size(res, 1);                  % 样本个数
%res = res(randperm(num_samples), :);         % 打乱数据集（不希望打乱时，注释该行）
num_train_s = round(num_size *num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度

%%  划分训练集和测试集
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);












% %% 改进麻雀ISSA参数设置
% 
% pop=3; % 种群数量
% Max_iter=5; % 最大迭代次数
% dim=3; % 优化LSTM的3个参数c
% lb = [50,50,0.001];%下边界
% ub = [300,300,0.01];%上边界
% numFeatures=f_;
% numResponses=outdim;
% fobj = @(x) fun(x,numFeatures,numResponses,res(:,end)) ;
% [Best_pos,Best_score,curve,BestNet]=SSA(pop,Max_iter,lb,ub,dim,fobj);
% 
% % 绘制进化曲线
% figure
% plot(curve,'r-','linewidth',3)
% xlabel('进化代数')
% ylabel('均方根误差RMSE')
% legend('最佳适应度')
% title('SSA-LSTM的进化收敛曲线')
% 
% disp('')
% disp(['最优隐藏单元数目为   ',num2str(round(Best_pos(1)))]);
% disp(['最优最大训练周期为   ',num2str(round(Best_pos(2)))]);
% disp(['最优初始学习率为   ',num2str((Best_pos(3)))]);


%%  数据平铺
%   将数据平铺成1维数据只是一种处理方式
%   也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
%   但是应该始终和输入层数据结构保持一致
p_train =  double(reshape(p_train, f_, 1, 1, M));
p_test  =  double(reshape(p_test , f_, 1, 1, N));
t_train =  double(t_train)';
t_test  =  double(t_test )';

%%  数据格式转换
for i = 1 : M
    Lp_train{i, 1} = p_train(:, :, 1, i);
end

for i = 1 : N
    Lp_test{i, 1}  = p_test( :, :, 1, i);
end
    
%%  建立模型
lgraph = layerGraph();                                                 % 建立空白网络结构

tempLayers = [
    sequenceInputLayer([f_, 1, 1], "Name", "sequence")                 % 建立输入层，输入数据结构为[f_, 1, 1]
    sequenceFoldingLayer("Name", "seqfold")];                          % 建立序列折叠层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = convolution2dLayer([1, 1], 32, "Name", "conv_1");   % 卷积层 卷积核[1, 1] 步长[1, 1] 通道数 32
  % batchNormalizationLayer("Name","batchnorm");                         批量归一化
lgraph = addLayers(lgraph,tempLayers);                                 % 将上述网络结构加入空白结构中
 
tempLayers = [
    reluLayer("Name", "relu_1")                                        % 激活层
    convolution2dLayer([1, 1], 64, "Name", "conv_2")                   % 卷积层 卷积核[1, 1] 步长[1, 1] 通道数 64
    reluLayer("Name", "relu_2")];                                      % 激活层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = [
    globalAveragePooling2dLayer("Name", "gapool")                      % 全局平均池化层
    fullyConnectedLayer(16, "Name", "fc_2")                            % SE注意力机制，通道数的1 / 4
    reluLayer("Name", "relu_3")                                        % 激活层
    fullyConnectedLayer(64, "Name", "fc_3")                            % SE注意力机制，数目和通道数相同
    sigmoidLayer("Name", "sigmoid")];                                  % 激活层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = multiplicationLayer(2, "Name", "multiplication");         % 点乘的注意力
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % 建立序列反折叠层
    flattenLayer("Name", "flatten")                                    % 网络铺平层
    lstmLayer(6, "Name", "lstm", "OutputMode", "last")                 % lstm层
     selfAttentionLayer(10,100,"Name","selfattention")                  %自注意力层
    fullyConnectedLayer(1, "Name", "fc")                               % 全连接层
    regressionLayer("Name", "regressionoutput")];                      % 回归层
lgraph = addLayers(lgraph, tempLayers);                                % 将上述网络结构加入空白结构中

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % 折叠层输出 连接 卷积层输入;
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
                                                                       % 折叠层输出 连接 反折叠层输入  
lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % 卷积层输出 链接 激活层
lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % 卷积层输出 链接 全局平均池化
lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % 激活层输出 链接 相乘层
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % 全连接输出 链接 相乘层
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % 点乘输出

%%  参数设置



options = trainingOptions('adam', ...   % adma 梯度下降算法
    'MaxEpochs',100, ...                % 最大训练次数 300
    'GradientThreshold',1,...           % 渐变的正阈值 1
    'ExecutionEnvironment','gpu',...    % 网络的执行环境 cpu
    'InitialLearnRate',0.01,...         % 初始学习率 0.01
    'LearnRateSchedule','none',...      % 训练期间降低整体学习率的方法 不降低
    'Shuffle','every-epoch',...         % 每次训练打乱数据集
    'SequenceLength',24,...             % 序列长度 24
    'Plots','training-progress',...     % 画出训练曲线
    'MiniBatchSize',8,...              % 训练批次大小 每次训练样本个数15
    'Verbose',0);                       % 有关训练进度的信息不打印到命令窗口中
               % 分析网络结构


%%  训练模型
net = trainNetwork(Lp_train, t_train, lgraph, options);

%%  模型预测
t_sim1 = predict(net, Lp_train);
t_sim2 = predict(net, Lp_test );

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1', ps_output);
T_sim2 = mapminmax('reverse', t_sim2', ps_output);
T_sim1=double(T_sim1);
T_sim2=double(T_sim2);
%%  显示网络结构
% analyzeNetwork(net)

%% 测试集结果
figure;
plotregression(T_test,T_sim2,['回归图']);
figure;
ploterrhist(T_test-T_sim2,['误差直方图']);
%%  均方根误差 RMSE
error1 = sqrt(sum((T_sim1 - T_train).^2)./M);
error2 = sqrt(sum((T_test - T_sim2).^2)./N);

%%
%决定系数
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

%%
%均方误差 MSE
mse1 = sum((T_sim1 - T_train).^2)./M;
mse2 = sum((T_sim2 - T_test).^2)./N;
%%
%RPD 剩余预测残差
SE1=std(T_sim1-T_train);
RPD1=std(T_train)/SE1;

SE=std(T_sim2-T_test);
RPD2=std(T_test)/SE;
%% 平均绝对误差MAE
MAE1 = mean(abs(T_train - T_sim1));
MAE2 = mean(abs(T_test - T_sim2));
%% 平均绝对百分比误差MAPE
% MAPE1 = mean(abs((T_train - T_sim1)./T_train));
% MAPE2 = mean(abs((T_test - T_sim2)./T_test));
%%  训练集绘图
% figure
% %plot(1:M,T_train,'r-*',1:M,T_sim1,'b-o','LineWidth',1)
% plot(1:M,T_train,'r-',1:M,T_sim1,'b-','LineWidth',1.5)
% legend('真实值','CNN-LSTM-Attention预测值')
% xlabel('预测样本')
% ylabel('预测结果')
% string={'训练集预测结果对比';['(R^2 =' num2str(R1) ' RMSE= ' num2str(error1) ' MSE= ' num2str(mse1) ' RPD= ' num2str(RPD1) ')' ]};
% title(string)
%% 预测集绘图
figure
plot(1:N,T_test,'r-',1:N,T_sim2,'b-','LineWidth',1.5)
legend('真实值','CNN-LSTM-Attention预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'测试集预测结果对比';['(R^2 =' num2str(R2) ' RMSE= ' num2str(error2)  ' MSE= ' num2str(mse2) ' RPD= ' num2str(RPD2) ')']};
title(string)

%% 测试集误差图
 figure  
 ERROR3=T_test-T_sim2;
 plot(T_test-T_sim2,'b-*','LineWidth',1.5)
 xlabel('测试集样本编号')
 ylabel('预测误差')
 title('测试集预测误差')
 grid on;
 legend('预测输出误差')
%% 绘制线性拟合图
%% 训练集拟合效果图
% figure
% plot(T_train,T_sim1,'*r');
% xlabel('真实值')
% ylabel('预测值')
% string = {'训练集效果图';['R^2_c=' num2str(R1)  '  RMSEC=' num2str(error1) ]};
% title(string)
% hold on ;h=lsline;
% set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
%% 预测集拟合效果图
figure
plot(T_test,T_sim2,'ob');
xlabel('真实值')
ylabel('预测值')
string1 = {'测试集效果图';['R^2_p=' num2str(R2)  '  RMSEP=' num2str(error2) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
%% 求平均
R3=(R1+R2)./2;
error3=(error1+error2)./2;
%% 总数据线性预测拟合图
% tsim=[T_sim1,T_sim2]';
% S=[T_train,T_test]';
% figure
% plot(S,tsim,'ob');
% xlabel('真实值')
% ylabel('预测值')
% string1 = {'所有样本拟合预测图';['R^2_p=' num2str(R3)  '  RMSEP=' num2str(error3) ]};
% title(string1)
% hold on ;h=lsline();
% set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
%% 打印出评价指标
disp(['-----------------------误差计算--------------------------'])
disp(['评价结果如下所示：'])
disp(['平均绝对误差MAE为：',num2str(MAE2)])
disp(['均方误差MSE为：       ',num2str(mse2)])
disp(['均方根误差RMSE为：  ',num2str(error2)])
disp(['决定系数R^2为：  ',num2str(R2)])
disp(['剩余预测残差RPD为：  ',num2str(RPD2)])
%disp(['平均绝对百分比误差MAPE为：  ',num2str(MAPE2)])
grid

%% 训练集和测试集的指标记录
ana=[R1,MAE1,error1,R2,MAE2,error2]; 
pre_result=[T_sim2',T_test'];
