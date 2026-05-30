% skip 1.5s
dt = 20e-6;      
skip_time = 1.5;   
start_idx = round(skip_time / dt) + 1;

Y_npc = z_pq_npc(start_idx:end, :); 
Y_pred = z_pq_pred(start_idx:end, :);
Y_t  = z_pq_t(start_idx:end, :);

N = min([size(Y_npc, 1), size(Y_pred, 1), size(Y_t, 1)]);
Y_npc = Y_npc(1:N, :);
Y_pred = Y_pred(1:N, :);
Y_t  = Y_t(1:N, :);

Y_mean  = mean(Y_npc, 1);                % 均值
Y_rms   = rms(Y_npc, 1);                 % 有效值 (RMS)
Y_range = max(Y_npc, [], 1) - min(Y_npc, [], 1); % 峰峰值 (极差)

RMSE_pred        = sqrt((1/N) * sum((Y_npc - Y_pred).^2, 1));
MAE_pred         = mean(abs(Y_npc - Y_pred), 1);
rRMSE_mean_pred  = RMSE_npc ./ abs(Y_mean);
rRMSE_rms_pred   = RMSE_npc ./ Y_rms;
NRMSE_range_pred = RMSE_npc ./ Y_range;

var_names = {'vcp', 'vcn', 'ia', 'ib', 'ic', 'p', 'q', 'vdc'};

fprintf('\n============================================= Error Analysis =============================================\n');
fprintf('                           [ PRED Error ]                       \n');
fprintf('----------------------------------------------------------------------------------------------------------\n');

for i = 1:size(Y_npc, 2)
    fprintf('【%s】:\n', upper(var_names{i}));
    fprintf('  - 绝对 RMSE   : %12.4f                        (物理单位)\n', RMSE_pred(i));
    fprintf('  - 绝对 MAE    : %12.4f                       (物理单位)\n', MAE_pred(i));
    fprintf('  - NRMSE (极差): %11.4f %%                     (占波动范围比例)\n', NRMSE_range_pred(i) * 100);
    fprintf('  - rRMSE (RMS) : %11.4f %%                     (占有效值比例)\n', rRMSE_rms_pred(i) * 100);
    fprintf('  - rRMSE (均值): %12.4f                       (>1因均值趋0)\n', rRMSE_mean_pred(i));
    fprintf('----------------------------------------------------------------------------------------------------------\n');
end

