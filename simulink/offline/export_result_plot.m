Y_IGBT = out.z_pq_true; 
Y_pred = out.z_pq_pred;
Y_SWF  = out.z_pq_swf;

fprintf('save .mat ...\n');
save('Y_IGBT.mat', 'Y_IGBT');
save('Y_pred.mat', 'Y_pred');
save('Y_SWF.mat', 'Y_SWF');
fprintf('saved Y_true.mat, Y_pred.mat, Y_swf.mat\n');