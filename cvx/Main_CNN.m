clear all
clc
a_n=[];
alpha=28.47446875;
beta=2.204250694;
N=100;
t_n=[];
Mmin=N*min(a_n.^2.*N.*t_n./beta);
Mmax=N*max(t_n);
step = 1;
cnt=1;
output = zeros(ceil((Mmax-Mmin)/step), N);
a_n=a_n';
t_n=t_n';
minans=zeros(ceil((Mmax-Mmin)/step), 1);
for i=1:ceil((Mmax-Mmin)/step)
    minans(i)=Inf;
end
m_record=zeros(ceil((Mmax-Mmin)/step), 1);
cluster = parcluster('local');  % 获取本地并行池的配置
cluster.NumWorkers = 32;  % 设置工作进程数量为 100，或适合你的需求的数量
pool = parpool(cluster);  % 创建并行池

parfor i = 1:ceil((Mmax - Mmin) / step)
    m = Mmin + (i - 1) * step;
    [solved, min_value] = mytest(t_n, a_n, alpha, beta, N, m);
    output(i, :) = solved;
    minans(i) = min_value;
    m_record(i) = m;
end

delete(pool);  % 关闭并行池

% ÕÒµ½ minans ÖÐµÄ×îÐ¡ÖµºÍ¶ÔÓ¦µÄË÷Òý
[min_minans, minans_idx] = min(minans);

% È¡ minans µÄ×îÐ¡Öµ¶ÔÓ¦µÄ output ÐÐ
min_output = output(minans_idx, :);
min_output=min_output';
finalmin=sum(alpha.*min_output./(N.*beta.*min_output-a_n.*a_n.*N.*N))*m_record(minans_idx);

% ½« min_output ±£´æµ½ CSV ÎÄ¼þ
csv_filename = 'CNN_output.csv';
csvwrite(csv_filename, min_output);

csv_filename = 'CNN_output_finalmin.csv';
csvwrite(csv_filename, finalmin);
