%% Common parameters
% Timestep (in seconds)
dt = 1e-4;

% Total simulation time (s)
tMax = 3;

% Stimulus onset time (s)
tOnLow = 0.500;
tOnHigh = 1.500;
tOff = 2.500;

% Range of time after onset to measure estimated odor concentrations (dt)
tRangeStart = cast(tOnHigh * (1 / dt), 'int32');
tRangeEnd = cast((tOnHigh + 1.000) * (1 / dt), 'int32');

% Times to check odor discrimination (dt)
tFastSniff = cast((tOnHigh + 0.100) * (1 / dt), 'int32');
tMedSniff = cast((tOnHigh + 0.200) * (1 / dt), 'int32');
tLongSniff = cast((tOnHigh + 1.000) * (1 / dt), 'int32');

% Cut-off accuracy for making plots
accuracy_threshold = 0.5;

% Color scheme
corderGeom = [0.1059    0.6196    0.4667
    0.8510    0.3725    0.0078
    0.4588    0.4392    0.7020];


%% Load data files
res_500 = load_results('remote/results_500');
res_1k = load_results('remote/results');
res_2k = load_results('remote/results_2k');
res_4k = load_results('remote/results_4k');
res_8k = load_results('remote/results_8k');



%% Plot thresholds
xs = [500 1000 2000, 4000, 8000];
res_all = cat(5, res_500, res_1k, res_2k, res_4k, res_8k);
all_times = [tFastSniff tMedSniff tLongSniff];
time_names = [" 100 ms" " 200 ms" " 1 s"];
time_save_names = ["fast" "med" "long"];
styles = [":", "--", "-"];

% figure('Position',[200,500,500,700],'WindowStyle','docked');
% hold on;

for t_idx = 1:3
    figure('Position',[200,500,500,700],'WindowStyle','docked');
    hold on;

    centers = zeros(3, size(res_all, 5));
    spreads = zeros(3, size(res_all, 5));
    n_iters = size(res_all, 1);
    
    for i = 1:size(res_all, 5)
        for j = 1:3
            vals = sum(res_all(:,j,all_times(t_idx)-tRangeStart,:,i) > accuracy_threshold, 4);
            centers(j,i) = mean(vals);
            spreads(j, i) = 1.96 * std(vals) / sqrt(n_iters);
        end
    end
    
    names = ["Naive" "Naive Dist." "Geometric"];

    for i = 1:3
        % errorbar(xs, centers(i,:), spreads(i,:), 'Color', corderGeom(i,:), 'DisplayName', strcat(names(i), time_names(t_idx)), 'LineStyle', styles(t_idx));
        fillBetween(centers(i,:), spreads(i,:), names(i), corderGeom(i,:))
    end
    % set(gca, 'XScale', 'log');

    
    xlabel('Max number of possible odors');
    ylabel('Max number of odor detected with >0.5 chance');
    title(strcat('Scaling: ', time_names(t_idx)));
    legend()

    saveas(gcf, strcat('fig/scaling', time_names(t_idx), '.fig'));
    hold off;
end

% saveas(gcf, 'fig/scaling.fig')
% hold off;



%% Functions
function [ results ] = load_results(directory)
resultsFiles = dir(fullfile(directory, 'thresh_*.mat'));

results = [];

for i = 1:length(resultsFiles)
    currentData = load(fullfile(directory, resultsFiles(i).name));
    results = [results; currentData.results];
end
end

function [ out ] = fillBetween(center, spread, name, corder)
    lwr = center - spread;
    upr = center + spread;
    xs = 1:size(center,2);

    p = patch([xs fliplr(xs)], [upr fliplr(lwr)], corder, 'EdgeColor', 'None', 'HandleVisibility', 'off');
    alpha(p, 0.25);
    out = plot(xs, center, 'DisplayName', name, 'Color', corder, 'LineWidth', 1.5);

    xticks(xs);
    xticklabels([500 1000 2000, 4000, 8000]);
end







