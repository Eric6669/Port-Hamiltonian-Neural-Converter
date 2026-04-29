clear;clc;
filename = 'RT_swf_20.mat'; % 'RT_ai_20.mat', 'RT_ai_40.mat', 'RT_swf_20.mat', 'RT_swf_40.mat', 'RT_IGBT_20.mat', 'RT_IGBT_40.mat'

S = load(filename);

vars = fieldnames(S);
data = S.(vars{1});
t = data(1, :);
valid_idx = (t <= 2.5);
Y_truncated = data(2:end, valid_idx).';

save('Y.mat', 'Y_truncated');
fprintf('\nsaved Y.mat！\n');
