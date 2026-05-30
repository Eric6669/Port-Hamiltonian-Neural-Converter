clear; clc;

file_list = {
    'HIL_ai_20.mat'
    'HIL_ai_40.mat'
    'HIL_swf_20.mat'
    'HIL_swf_40.mat'
    'HIL_IGBT_20.mat'
    'HIL_IGBT_40.mat'
};

for k = 1:numel(file_list)

    filename = file_list{k};

    if ~isfile(filename)
        continue;
    end

    S = load(filename);

    vars = fieldnames(S);
    data = S.(vars{1});

    t = data(1, :);
    valid_idx = (t <= 2.5);

    Y_truncated = data(2:end, valid_idx).';

    [~, name, ~] = fileparts(filename);
    name = erase(name, 'HIL_');
    save_name = ['Y_' name '.mat'];

    save(save_name, 'Y_truncated');

    fprintf('Processed: %-20s -> %s\n', filename, save_name);
end

fprintf('\nAll files processed!\n');