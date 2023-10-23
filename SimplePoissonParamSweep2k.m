%% Set neuron parameters

% Neuron time constants (in seconds)
tauE = 0.020;
tauI = 0.030;

% Number of sensors
nSens = 300;
% nSens = 25;

% Number of odors
nOdor = 2000;
% nOdor = 50;

% Number of granule cells
nGranule = 5*nOdor;

% Baseline activity
r0 = 1;

% High and low concentrations
cLow = 10;
cHigh = 40;
cPredHigh = 20;

% Laplace prior strength
lambda = 1;

% Regularization strength
a = 0.5;

% Flag indicating whether to use std or max norm
maxFlag = true;

scaleSqrt = true;

%% Set simulation parameters
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



%% Set plotting options

corderLowHigh = lines(3);
corderGeom = [0.1059    0.6196    0.4667
    0.8510    0.3725    0.0078
    0.4588    0.4392    0.7020];


%% Parameter sweep
disp("ID:");
rng(12347*datetime("now").Second);
id = num2str(randi(9999999));
% fprintf('Using ID=%10d\n', id);
disp(id);

n_iters = 1;
ns = 1:100;
results = zeros(n_iters, 3, tRangeEnd - tRangeStart, length(ns));
results_compk = zeros(n_iters, 3, tRangeEnd - tRangeStart, length(ns));

for i = 1:n_iters
    fprintf('ITER = %2d\n', i);

    for nHigh = ns
        % Generate the affinity matrix
        % Random affinity matrix
        A = gamrnd(0.37,0.36,nSens,nOdor);
        A = A ./ max(A, [], 2);
        
        % A=double(rand(nSens,nOdor)<0.15);
        
        % A = A / max(abs(A(:)));
        
        % Generate the projection matrices
        Q = orth(randn(nOdor, nGranule)')';
        % Q = eye(nOdor);
        % Q = sprand(nOdor, nGranule, 0.15);
        
        C = A'*A;
        C = C ./ trace(C) * nOdor;
        
        [UC, SC] = eig(C);
        
        
        sInv = diag(1./sqrt(diag(SC) + a));
        B = UC * sInv * UC';
        
        Gnaive = eye(nOdor);
        Gdist = Q;
        Ggeom = B * Q;
        
        % Precompute required matrix products
        AGdist = A*Gdist;
        AGgeom = A*Ggeom;
        
        % Compute normalization factors
        if maxFlag
            normConstNaive = max(abs(A(:)));
            normConstDist = max(abs(AGdist(:)));
            normConstGeom = max(abs(AGgeom(:)));
        else
            normConstNaive = std(A(:));
            normConstDist = std(AGdist(:));
            normConstGeom = std(AGgeom(:));
        end

        if scaleSqrt
            C = 50;
            normConstNaive = normConstNaive * (sqrt(nGranule) / C);
            normConstDist = normConstDist * (sqrt(nGranule) / C);
            normConstGeom = normConstGeom * (sqrt(nGranule) / C);
        end
        
        AGnaive = A / normConstNaive;
        AGdist = AGdist / normConstDist;
        AGgeom = AGgeom  / normConstGeom;
        
        Gnaive = Gnaive / normConstNaive;
        Gdist = Gdist / normConstDist;
        Ggeom = Ggeom / normConstGeom;

        fprintf('nHigh = %2d\n', nHigh);
        % Generate the sensory scene
        [ sMat, cMatTrue, nT] = generateSensoryScene(nHigh, dt, tMax, tOnLow, tOnHigh, tOff, nOdor, r0, cLow, cHigh, A);
        
        % Run the naive model
        [ rExcNaive, rInhNaive, cEstNaive ] = integratePoissonSamplingCircuitNoiseless(sMat, nT, dt, nSens, nOdor, r0, tauE, tauI, lambda, AGnaive, Gnaive);
        
        % Run the naive distributed model
        [ rExcDist, rInhDist, cEstDist ] = integratePoissonSamplingCircuitNoiseless(sMat, nT, dt, nSens, nGranule, r0, tauE, tauI, lambda, AGdist, Gdist);
        
        % Run the geometry-aware model
        [ rExcGeom, rInhGeom, cEstGeom ] = integratePoissonSamplingCircuitNoiseless(sMat, nT, dt, nSens, nGranule, r0, tauE, tauI, lambda, AGgeom, Ggeom);
        
        cEstAll = cat(3, cEstNaive, cEstDist, cEstGeom);

        for t = 1:(tRangeEnd-tRangeStart)
            for j = 1:3
                detections = cEstAll(t+tRangeStart,:,j) > cPredHigh;
                nCorrect = sum(detections(1:nHigh));
                results(i, j, t, nHigh) = nCorrect / nHigh;
        
                [~, top_k] = maxk(cEstAll(t+tRangeStart,:,j), nHigh);
                nCorrect = sum(top_k <= nHigh);
                results_compk(i, j, t, nHigh) = nCorrect / nHigh;
            end
        end
    end
end


save(['results_2k/thresh_' id '.mat'], 'results');
save(['results_2k/compk_' id '.mat'], 'results_compk');


%% Load saved files
directory = 'remote/results_2k';
resultsFiles = dir(fullfile(directory, 'thresh_*.mat'));
resultsFiles_compk = dir(fullfile(directory, 'compk_*.mat'));

results = [];
results_compk = [];

for i = 1:length(resultsFiles)
    currentData = load(fullfile(directory, resultsFiles(i).name));
    results = [results; currentData.results];

    currentData = load(fullfile(directory, resultsFiles_compk(i).name));
    results_compk = [results_compk; currentData.results_compk];
end

%% Plot estimation accuracy at different times
% results = load('remote/results.mat').results;
n_iters = size(results,1);
fig_ext = '.fig';

names = ["Naive" "Naive Dist." "Geometric"];

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;

for i = 1:3
    curr = results(:,i,tFastSniff-tRangeStart,:);
    center = squeeze(mean(curr, 1))';
    spread = squeeze(std(curr, 1))' * (1.96 / sqrt(n_iters));

    fillBetween(center, spread, names(i), corderGeom(i,:));
end


ylim([0 1.01]);
xlabel('number of odors');
ylabel('proportion estimated correct');
title('After 100 ms')
legend()
saveas(gcf, [ 'fig/prop_correct_fast' fig_ext])
hold off;


figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;

for i = 1:3
    curr = results(:,i,tMedSniff-tRangeStart,:);
    center = squeeze(mean(curr, 1))';
    spread = squeeze(std(curr, 1))' * (1.96 / sqrt(n_iters));

    fillBetween(center, spread, names(i), corderGeom(i,:));
end

ylim([0 1.01]);
xlabel('number of odors');
ylabel('proportion estimated correct');
title('After 200 ms')
legend()
saveas(gcf, [ 'fig/prop_correct_med' fig_ext])
hold off;


figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;

for i = 1:3
    curr = results(:,i,tLongSniff-tRangeStart,:);
    center = squeeze(mean(curr, 1))';
    spread = squeeze(std(curr, 1))' * (1.96 / sqrt(n_iters));

    fillBetween(center, spread, names(i), corderGeom(i,:));
end

ylim([0 1.01]);
xlabel('number of odors');
ylabel('proportion estimated correct');
title('After 1 s')
legend()
saveas(gcf, [ 'fig/prop_correct_long' fig_ext])
hold off;


% top-K plots
% ------------------------------------------------------------------------------------------------
figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;

for i = 1:3
    curr = results_compk(:,i,tFastSniff-tRangeStart,:);
    center = squeeze(mean(curr, 1))';
    spread = squeeze(std(curr, 1))' * (1.96 / sqrt(n_iters));

    fillBetween(center, spread, names(i), corderGeom(i,:));
end

ylim([0 1.01]);
xlabel('number of odors');
ylabel('proportion estimated correct');
title('After 100 ms')
legend()
saveas(gcf, [ 'fig/prop_correct_fast_compk' fig_ext])
hold off;


figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;

for i = 1:3
    curr = results_compk(:,i,tMedSniff-tRangeStart,:);
    center = squeeze(mean(curr, 1))';
    spread = squeeze(std(curr, 1))' * (1.96 / sqrt(n_iters));

    fillBetween(center, spread, names(i), corderGeom(i,:));
end

ylim([0 1.01]);
xlabel('number of odors');
ylabel('proportion estimated correct');
title('After 200 ms')
legend()
saveas(gcf, [ 'fig/prop_correct_med_compk' fig_ext])
hold off;


figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;

for i = 1:3
    curr = results_compk(:,i,tLongSniff-tRangeStart,:);
    center = squeeze(mean(curr, 1))';
    spread = squeeze(std(curr, 1))' * (1.96 / sqrt(n_iters));

    fillBetween(center, spread, names(i), corderGeom(i,:));
end

ylim([0 1.01]);
xlabel('number of odors');
ylabel('proportion estimated correct');
title('After 1 s')
legend()
saveas(gcf, [ 'fig/prop_correct_long_compk' fig_ext])
hold off;

%% Plot heatmaps of estimated accuracy
results_m = squeeze(mean(results(:,:,:,:), 1));
results_compk_m = squeeze(mean(results_compk(:,:,:,:), 1));

intv = cast(0.01 * (1 / dt), 'int32');

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
p = pcolor(squeeze(results_m(1,1:intv:length(results),:))');
set(p, 'EdgeColor', 'None');
colorbar();
A = squeeze(results_m(1,1:intv:length(results),:))';
A = imgaussfilt(A, 4);
contour(A,'EdgeColor', 'k', 'ShowText', 'on');
hold off;


xlabel('time (per 10ms after odor onset)')
ylabel('number of odors')
title('Naive')
saveas(gcf, [ 'fig/heatmap_naive' fig_ext]);

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
p = pcolor(squeeze(results_m(2,1:intv:length(results),:))');
set(p, 'EdgeColor', 'None');
colorbar();
A = squeeze(results_m(2,1:intv:length(results),:))';
A = imgaussfilt(A, 4);
contour(A, 'EdgeColor', 'k', 'ShowText', 'on');
hold off;

xlabel('time (per 10ms after odor onset)')
ylabel('number of odors')
title('Naive Distributed')
saveas(gcf, [ 'fig/heatmap_naive_dist' fig_ext])

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
p = pcolor(squeeze(results_m(3,1:intv:length(results),:))');
set(p, 'EdgeColor', 'None');
colorbar();
A = squeeze(results_m(3,1:intv:length(results),:))';
A = imgaussfilt(A, 4);
contour(A, 'EdgeColor', 'k', 'ShowText', 'on');
hold off;

xlabel('time (per 10ms after odor onset)')
ylabel('number of odors')
title('Geometric')
saveas(gcf, [ 'fig/heatmap_geom' fig_ext])

% top-K plots
% ------------------------------------------------------------------------------------------------
figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
p = pcolor(squeeze(results_compk_m(1,1:intv:length(results),:))');
set(p, 'EdgeColor', 'None');
colorbar();
A = squeeze(results_compk_m(1,1:intv:length(results),:))';
A = imgaussfilt(A, 4);
contour(A, 'EdgeColor', 'k', 'ShowText', 'on');
hold off;

xlabel('time (per 10ms after odor onset)')
ylabel('number of odors')
title('Naive')
saveas(gcf, [ 'fig/heatmap_naive_compk' fig_ext])

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
p = pcolor(squeeze(results_compk_m(2,1:intv:length(results),:))');
set(p, 'EdgeColor', 'None');
colorbar();
A = squeeze(results_compk_m(2,1:intv:length(results),:))';
A = imgaussfilt(A, 4);
contour(A, 'EdgeColor', 'k', 'ShowText', 'on');
hold off;

xlabel('time (per 10ms after odor onset)')
ylabel('number of odors')
title('Naive Distributed')
saveas(gcf, [ 'fig/heatmap_naive_dist_compk' fig_ext])

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
p = pcolor(squeeze(results_compk_m(3,1:intv:length(results),:))');
set(p, 'EdgeColor', 'None');
colorbar();
A = squeeze(results_compk_m(3,1:intv:length(results),:))';
A = imgaussfilt(A, 4);
contour(A, 'EdgeColor', 'k', 'ShowText', 'on');
hold off;

xlabel('time (per 10ms after odor onset)')
ylabel('number of odors')
title('Geometric')
saveas(gcf, [ 'fig/heatmap_geom_compk' fig_ext])

%}
%% Function definitions
function [sMat, cMatTrue, nT] = generateSensoryScene(nHigh, dt, tMax, tOnLow, tOnHigh, tOff, nOdor, r0, cLow, cHigh, A)
nLow = 0;

% Time vector
t = (0:dt:tMax-dt)';
nT = length(t);

% First, check that onset and offset times are ordered
if tOnLow > tOnHigh || tOnLow > tOff || tOnHigh > tOff || tOff > tMax
    error('Invalid onset and offset times')
end

% Assuming the ordering above, generate a matrix of odor concentrations
% Size will be 4 x nOdor
cMat = [zeros(1, nOdor); [cLow * ones(1,nLow), zeros(1, nOdor - nLow)]; [cLow * ones(1,nLow), cHigh * ones(1,nHigh), zeros(1, nOdor - nLow - nHigh)]; zeros(1, nOdor)];

% Form the Poisson rate matrix (4 x nSens)
lambdaMat = r0 + cMat * A';

% Sample from a Poisson distribution to get sensor activities (4 x nSens)
pMat = poissrnd(lambdaMat);

% Expand the Poisson matrix in time (nT x nSens)
sMat = (t < tOnLow) * pMat(1,:) + (t >= tOnLow & t < tOnHigh) * pMat(2,:) + (t >= tOnHigh & t < tOff) * pMat(3,:) + (t >= tOff) * pMat(4,:);

% Generate a matrix of the true concentrations
cMatTrue = [cLow * double(t >= tOnLow & t <= tOff) * ones(1, nLow), cHigh * double(t >= tOnHigh & t <= tOff) * ones(1, nHigh), zeros(nT, nOdor - nLow - nHigh)];
end



function [ rExc, rInh, cEst ] = integratePoissonSamplingCircuitNoiseless(sMat, nT, dt, nSens, nGranule, r0, tauE, tauI, lambda, AG, G)
rExc = nan(nT, nSens);
rInh = nan(nT, nGranule);

rExc(1,:) = 1./r0;
rInh(1,:) = 0;

% Integrate the model using Euler-Maruyama
for indT = 2:nT

    drExc = sMat(indT, :) - rExc(indT-1,:) .* (r0 + rInh(indT-1,:) * AG');
    drInh = (rExc(indT-1,:) - 1) * AG - lambda * sign(rInh(indT-1,:) * G') * G;

    rExc(indT,:) = rExc(indT-1,:) + dt * drExc / tauE;
    rInh(indT,:) = rInh(indT-1,:) + dt * drInh / tauI;

end

% Compute the concentration estimate
cEst = rInh * G';

end

function [ out ] = fillBetween(center, spread, name, corder)
    lwr = center - spread;
    upr = center + spread;
    xs = 1:size(center,2);

    p = patch([xs fliplr(xs)], [upr fliplr(lwr)], corder, 'EdgeColor', 'None', 'HandleVisibility', 'off');
    alpha(p, 0.25);
    out = plot(xs, center, 'DisplayName', name, 'Color', corder, 'LineWidth', 1.5);
end

