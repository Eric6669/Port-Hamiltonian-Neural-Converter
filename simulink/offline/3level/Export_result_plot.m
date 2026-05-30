Y_NPC = z_pq_npc; 
Y_pred = z_pq_pred;
Y_T  = z_pq_t;

fprintf('save .mat ...\n');
save('Y_NPC.mat', 'Y_NPC');
save('Y_pred.mat', 'Y_pred');
save('Y_T.mat', 'Y_T');
fprintf('saved Y_NPC.mat, Y_pred.mat, Y_T.mat\n');