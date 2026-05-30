% skip 1.5s
dt = 20e-6;      
skip_time = 1.5;   
start_idx = round(skip_time / dt) + 1;

Y      = out.z_pq_true(start_idx:end, :); 
Y_pred = out.z_pq_pred(start_idx:end, :);
Y_swf  = out.z_pq_swf(start_idx:end, :);

N = min([size(Y, 1), size(Y_pred, 1), size(Y_swf, 1)]);
Y      = Y(1:N, :);
Y_pred = Y_pred(1:N, :);
Y_swf  = Y_swf(1:N, :);

Y_mean  = mean(Y, 1);                % 均值
Y_rms   = rms(Y, 1);                 % 有效值 (RMS)
Y_range = max(Y, [], 1) - min(Y, [], 1); % 峰峰值 (极差)

RMSE_pred        = sqrt((1/N) * sum((Y - Y_pred).^2, 1));
MAE_pred         = mean(abs(Y - Y_pred), 1);
rRMSE_mean_pred  = RMSE_pred ./ abs(Y_mean);
rRMSE_rms_pred   = RMSE_pred ./ Y_rms;
NRMSE_range_pred = RMSE_pred ./ Y_range;

RMSE_swf        = sqrt((1/N) * sum((Y - Y_swf).^2, 1));
MAE_swf         = mean(abs(Y - Y_swf), 1);
rRMSE_mean_swf  = RMSE_swf ./ abs(Y_mean);
rRMSE_rms_swf   = RMSE_swf ./ Y_rms;
NRMSE_range_swf = RMSE_swf ./ Y_range;

var_names = {'vcp', 'vcn', 'ia', 'ib', 'ic', 'p', 'q', 'vdc'};

fprintf('\n============================================= Error Analysis =============================================\n');
fprintf('                           [ PRED Error ]            |            [ SWF Error ]\n');
fprintf('----------------------------------------------------------------------------------------------------------\n');

for i = 1:size(Y, 2)
    fprintf('【%s】:\n', upper(var_names{i}));
    fprintf('  - 绝对 RMSE   : %12.4f                       | %12.4f         (物理单位)\n', RMSE_pred(i), RMSE_swf(i));
    fprintf('  - 绝对 MAE    : %12.4f                       | %12.4f         (物理单位)\n', MAE_pred(i), MAE_swf(i));
    fprintf('  - NRMSE (极差): %11.4f %%                     | %11.4f %%       (占波动范围比例)\n', NRMSE_range_pred(i) * 100, NRMSE_range_swf(i) * 100);
    fprintf('  - rRMSE (RMS) : %11.4f %%                     | %11.4f %%       (占有效值比例)\n', rRMSE_rms_pred(i) * 100, rRMSE_rms_swf(i) * 100);
    fprintf('  - rRMSE (均值): %12.4f                       | %12.4f         (>1因均值趋0)\n', rRMSE_mean_pred(i), rRMSE_mean_swf(i));
    fprintf('----------------------------------------------------------------------------------------------------------\n');
end

