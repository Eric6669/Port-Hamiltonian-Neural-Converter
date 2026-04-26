% Compare pH-NODE and switching-function outputs against the IGBT/Diode reference.
% The first 0.5 s are skipped to remove initialization transients.

dt = 20e-6;
skip_time = 0.5;
start_idx = round(skip_time / dt) + 1;

Y      = out.z_pq_true(start_idx:end, :);
Y_pred = out.z_pq_pred(start_idx:end, :);
Y_swf  = out.z_pq_swf(start_idx:end, :);

N = min([size(Y, 1), size(Y_pred, 1), size(Y_swf, 1)]);
Y      = Y(1:N, :);
Y_pred = Y_pred(1:N, :);
Y_swf  = Y_swf(1:N, :);

Y_mean  = mean(Y, 1);
Y_rms   = rms(Y, 1);
Y_range = max(Y, [], 1) - min(Y, [], 1);

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

fprintf('
============================================= Error Analysis =============================================
');
fprintf('                           [ pH-NODE Error ]          |      [ Switching Function Error ]
');
fprintf('----------------------------------------------------------------------------------------------------------
');

for i = 1:size(Y, 2)
    fprintf('[%s]
', upper(var_names{i}));
    fprintf('  - RMSE         : %12.4f                       | %12.4f
', RMSE_pred(i), RMSE_swf(i));
    fprintf('  - MAE          : %12.4f                       | %12.4f
', MAE_pred(i), MAE_swf(i));
    fprintf('  - NRMSE(range) : %11.4f %%                     | %11.4f %%
', NRMSE_range_pred(i) * 100, NRMSE_range_swf(i) * 100);
    fprintf('  - rRMSE(RMS)   : %11.4f %%                     | %11.4f %%
', rRMSE_rms_pred(i) * 100, rRMSE_rms_swf(i) * 100);
    fprintf('  - rRMSE(mean)  : %12.4f                       | %12.4f
', rRMSE_mean_pred(i), rRMSE_mean_swf(i));
    fprintf('----------------------------------------------------------------------------------------------------------
');
end
