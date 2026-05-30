%% Seed
rng(77);

%% Set
num_sims = 200;
sim_time = 3.5;
update_interval = 0.15;
t_start = 0.5;

Ts = 20e-6;
t_vec = (0:Ts:sim_time)'; 

%% Base
Vdc_base = 3e3;
fs_base = 60;
Vref_base = Vdc_base/2;
Vac_peak_MV_base = Vref_base*sqrt(2/3);

%% Mdl
mdl = 'Parsim_for_3level';    
load_system(mdl);
in(1:num_sims) = Simulink.SimulationInput(mdl);


%% Parasim
for i = 1:num_sims

    num_intervals = ceil((sim_time - t_start) / update_interval); 

    % sin para
    freqs_P = 1+99*rand(num_intervals,1);
    freqs_Q = 1+99*rand(num_intervals,1);
    freqs_Vdc = 1+99*rand(num_intervals,1);
    freqs_Vac = 1+99*rand(num_intervals, 1);

    amps_P = -1+2*rand(num_intervals,1);
    amps_Q = -1+2*rand(num_intervals,1);
    amps_Vdc = -1+2*rand(num_intervals,1);
    amps_Vac = -1+2*rand(num_intervals, 1);

    Pset_arr = -0.8*ones(size(t_vec)); 
    Qset_arr = 0.0*ones(size(t_vec));
    Vdc_arr  = Vdc_base * ones(size(t_vec)); 
    Vac_envelope_arr = Vac_peak_MV_base * ones(size(t_vec));
    
    % chosen variables [Pset Qset Vdc Vac]
    target_vars = randi([1, 4], num_intervals, 1);

    % disturb
    for k = 1:length(t_vec)
        t = t_vec(k);

        P_current = -0.8;
        Q_current = 0.0;
        Vdc_current = Vdc_base;
        Vac_current = Vac_peak_MV_base;

        if t >= t_start
            idx = floor((t - t_start) / update_interval) + 1;
            if idx > num_intervals; idx = num_intervals; end
            
            target = target_vars(idx);

            % ------------------------------------------------------
            %  P +/-0.4, Q +/-0.4, Vdc +/-10%, Vac +/-15%
            % ------------------------------------------------------
            if target == 1
                P_current = -0.8 + 0.4 * amps_P(idx) * sin(2*pi*freqs_P(idx)*(t - t_start));
            elseif target == 2
                Q_current = 0.0 + 0.4 * amps_Q(idx) * sin(2*pi*freqs_Q(idx)*(t - t_start));
            elseif target == 3
                Vdc_current = Vdc_base * (1 + 0.1 * amps_Vdc(idx) * sin(2*pi*freqs_Vdc(idx)*(t - t_start)));
            elseif target == 4
                Vac_current = Vac_peak_MV_base * (1 + 0.15 * amps_Vac(idx) * sin(2*pi*freqs_Vac(idx)*(t - t_start)));
            end
            
            % record
            Pset_arr(k) = P_current;
            Qset_arr(k) = Q_current;
            Vdc_arr(k)  = Vdc_current;
            Vac_envelope_arr(k) = Vac_current;
        end
    end

    % Three-phase voltage
    omega = 2 * pi * fs_base;
    Va_arr = Vac_envelope_arr .* sin(omega * t_vec);
    Vb_arr = Vac_envelope_arr .* sin(omega * t_vec - 2*pi/3);
    Vc_arr = Vac_envelope_arr .* sin(omega * t_vec + 2*pi/3);
    
    % Timeseries
    ts_Pset = timeseries(Pset_arr, t_vec);
    ts_Qset = timeseries(Qset_arr, t_vec);
    ts_Vdc  = timeseries(Vdc_arr, t_vec);

    ts_Va = timeseries(Va_arr, t_vec);
    ts_Vb = timeseries(Vb_arr, t_vec);
    ts_Vc = timeseries(Vc_arr, t_vec);

    % fill in
    in(i) = in(i).setModelParameter('StopTime', num2str(sim_time));
    
    in(i) = in(i).setVariable('Pset', ts_Pset);
    in(i) = in(i).setVariable('Qset', ts_Qset);
    in(i) = in(i).setVariable('Vdc_set', ts_Vdc);

    in(i) = in(i).setVariable('Va_set', ts_Va);
    in(i) = in(i).setVariable('Vb_set', ts_Vb);
    in(i) = in(i).setVariable('Vc_set', ts_Vc);
end

disp('Parasim...')
out = parsim(in, 'ShowProgress', 'on');

%% Save
if ~exist('raw', 'dir')
    mkdir('raw');
end

success_count = 0;
start_idx = round(t_start / Ts) + 1;

for i = 1:num_sims
    if out(i).ErrorMessage == "" 
        success_count = success_count + 1;

        raw_data = out(i).sim_data; 
        clean_data = raw_data(start_idx:end, :);
        filename = sprintf('raw/sim_record_%03d.mat', i);
        save(filename, 'clean_data');
    else
        warning(['No.', num2str(i), 'simulation collapse', out(i).ErrorMessage]);
    end
end
disp([num2str(success_count), 'saved']);
