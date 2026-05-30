Ts=20e-6; % system time step
Tctrl=Ts; % controller time step

Pref=14e3;  % nomial power of machine %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 0.3e6 
Vdc=720; % dc voltage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 3e3 
Vref=Vdc/2;  % Induction machine rms L-L voltage

% C=3e-6;
C1=3300e-6; % RC Branch C
R1=1e-3; % RC Branch R

Vac_peak_MV=220*sqrt(2); % ac phase peak MV side %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Vref*sqrt(2/3)
fs=50;  % grid frequency %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 60 

Rac_MV=0.1; % mvac source ressitor
Lac_MV=0.005*3; % LVac source inductance; was 0.005


f_cari=2e4/4.1; %2e4/4.1
Ron=0.001;  %0.001
Rsim=5e7;   %5e7;
Csim=2.5e-9;
IGBT_Ron = 1e-3;%1e-3;
IGBT_Rs = 1e5;  %original 1e5
IGBT_Cs = inf;

DRsnb=500;           %snubber R
DCsnb=2.5000e-09;    %snubber C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ffiltragemes=60; 
% Lm=34.7e-3; 
% Llr=0.8e-3; 
% Rr=0.228; 
% Pm=2; 
% Rs=0.087;
% 
% % Switches Parameters
% 
% Diode_Ron = 1e-3;
% T_deadtime=5e-6;
% R_Snubber = 1000;
% C_Snubber = 0.01;
% V_opal=Simulink.Variant('V_MODEslv==0');
% V_sps=Simulink.Variant('V_MODEslv==1');
% V_MODEslv=0;
% 
% V_2lvl=Simulink.Variant('V_model==0');
% V_npc=Simulink.Variant('V_model==1');
% V_ttype=Simulink.Variant('V_model==2');
% V_model=0;
% 
% V_NoGND=Simulink.Variant('V_GND1==0');
% V_GND=Simulink.Variant('V_GND1==1');
% V_GND1=0;