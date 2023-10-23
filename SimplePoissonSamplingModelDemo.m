%% Set neuron parameters

% Neuron time constants (in seconds)
tauE = 0.020;
tauI = 0.030;

% Number of sensors
nSens = 300;

% Number of odors
nOdor = 1000;

% Number of granule cells
nGranule = 5*nOdor;

% Baseline activity
r0 = 1;

% Number of high and low odors
nLow = 5;
nHigh = 5;

% High and low concentrations
cLow = 10;
cHigh = 40;

% Laplace prior strength
lambda = 1;

% Regularization strength
a = 0.5;

% Flag indicating whether to use std or max norm
normFlag = 'max';

%% Set simulation parameters

% Timestep (in seconds)
dt = 1e-5;

% Total simulation time (s)
tMax = 3;

% Stimulus onset time (s)
tOnLow = 0.500;
tOnHigh = 1.500;
tOff = 2.500;

%% Set baseline simulation parameters

dtMALA = 1e-5;
% nMALA = 1e9;
nMALA = 1e8;

% nBurn = 1e8;
nBurn = 1e7;

%% Set plotting options

corderLowHigh = lines(3);
corderGeom = [0.1059    0.6196    0.4667;
    0.8510    0.3725    0.0078;
    0.4588    0.4392    0.7020;
    0, 0, 0];

%% Generate the affinity matrix

% Random affinity matrix
A = gamrnd(0.37,0.36,nSens,nOdor);
A = A ./ max(A, [], 2);

% A=double(rand(nSens,nOdor)<0.15);

% A = A / max(abs(A(:)));

%% Generate the projection matrices

Q = orth(randn(nOdor, nGranule)')';
% Q = eye(nOdor);
% Q = sprand(nOdor, nGranule, 0.05);

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
switch normFlag
    case 'max'
        normConstNaive = max(abs(A(:)));
        normConstDist = max(abs(AGdist(:)));
        normConstGeom = max(abs(AGgeom(:)));
    case 'std'
        normConstNaive = std(A(:));
        normConstDist = std(AGdist(:));
        normConstGeom = std(AGgeom(:));
    case 'abs'
        normConstNaive = mean(abs(A(:)));
        normConstDist = mean(abs(AGdist(:)));
        normConstGeom = mean(abs(AGgeom(:)));
    otherwise
        error('Invalid normalization flag: %s.', normFlag);
end

AGnaive = A / normConstNaive;
AGdist = AGdist / normConstDist;
AGgeom = AGgeom  / normConstGeom;

Gnaive = Gnaive / normConstNaive;
Gdist = Gdist / normConstDist;
Ggeom = Ggeom / normConstGeom;


%% Generate the sensory scene

tic;

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

fprintf('Generated sensory scene in %f seconds\n', toc);


%% Run the naive model

[ rExcNaive, rInhNaive, cEstNaive ] = IntegratePoissonSamplingCircuit(sMat, nT, dt, nSens, nOdor, r0, tauE, tauI, lambda, AGnaive, Gnaive);

fprintf('Completed naive simulation in %f seconds.\n', toc);

%% Run the naive distributed model

[ rExcDist, rInhDist, cEstDist ] = IntegratePoissonSamplingCircuit(sMat, nT, dt, nSens, nGranule, r0, tauE, tauI, lambda, AGdist, Gdist);

fprintf('Completed naive distributed simulation in %f seconds.\n', toc);

%% Run the geometry-aware model

[ rExcGeom, rInhGeom, cEstGeom ] = IntegratePoissonSamplingCircuit(sMat, nT, dt, nSens, nGranule, r0, tauE, tauI, lambda, AGgeom, Ggeom);

fprintf('Completed geometry-aware simulation in %f seconds.\n', toc);

%% Compute the MAP estimates

opts = optimoptions('fmincon','display','final', 'MaxFunctionEvaluations', 1e5);

sLow = sMat(find(t>tOnLow, 1, 'first'),:);
mapFun = @(c) A'*(sLow' ./ (r0+A*c) - 1) - lambda;
mapLow = fmincon(@(c) vecnorm(mapFun(c)).^2, rand(nOdor,1), [], [], [], [], zeros(nOdor,1), 100*ones(nOdor,1), [],opts);

sHigh = sMat(find(t>tOnHigh, 1, 'first'),:);
mapFun = @(c) A'*(sHigh' ./ (r0+A*c) - 1) - lambda;
mapHigh = fmincon(@(c) vecnorm(mapFun(c)).^2, rand(nOdor,1), [], [], [], [], zeros(nOdor,1), 100*ones(nOdor,1), [],opts);

%% Compute baseline posterior samples

tic;
sLow = sMat(find(t>tOnLow, 1, 'first'),:);
[cLowMean, cLowVar] = ComputeBaselineMeanAndVarianceLangevin(dtMALA, nMALA, nBurn, nOdor, nSens, A, lambda, r0, sLow, B);
fprintf('Completed low concentration baseline Langevin simulation in %f seconds.\n', toc);

tic;
sHigh = sMat(find(t>tOnHigh, 1, 'first'),:);
[cHighMean, cHighVar] = ComputeBaselineMeanAndVarianceLangevin(dtMALA, nMALA, nBurn, nOdor, nSens, A, lambda, r0, sHigh, B);
fprintf('Completed high concentration baseline Langevin simulation in %f seconds.\n', toc);


%% Downsample and smooth data for plotting

dtPlot = 0.001;

avgSpan = round(0.100/dtPlot);

tPlot = (0:dtPlot:tMax-dtPlot)';

cLowPlot = interp1(t, cMatTrue(:,1), tPlot);
cHighPlot = interp1(t, cMatTrue(:,nLow+1), tPlot);

cNaivePlot = interp1(t, cEstNaive, tPlot);
cDistPlot = interp1(t, cEstDist, tPlot);
cGeomPlot = interp1(t, cEstGeom, tPlot);

%% Make a plot of weight matrix entries

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
histogram(AGnaive(:), 'EdgeColor','None', 'FaceColor',corderGeom(1,:));
histogram(AGdist(:), 'EdgeColor','None', 'FaceColor',corderGeom(2,:));
histogram(AGgeom(:), 'EdgeColor','None', 'FaceColor',corderGeom(3,:));
xlabel('(A\Gamma)_{ij})')
ylabel('count');
axis('square');
set(gca, 'box','off','linewidth',2,'fontsize', 16);

%% Plot the results

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
if nLow + nHigh < nOdor
    bkgMean = mean(fast_moving_average(cNaivePlot(:,nLow+nHigh+1:end), avgSpan),2);
    bkgSD = std(fast_moving_average(cNaivePlot(:,nLow+nHigh+1:end), avgSpan),0,2);
    PlotAsymmetricErrorPatch(tPlot, bkgMean, bkgSD, bkgSD, [0.5,0.5,0.5]);
end
plot(tPlot, fast_moving_average(cNaivePlot(:,1:nLow), avgSpan), '-', 'Color', corderLowHigh(1,:), 'LineWidth', 2);
plot(tPlot, fast_moving_average(cNaivePlot(:,nLow+1:nLow+nHigh), avgSpan), '-', 'Color', corderLowHigh(2,:), 'LineWidth', 2);
plot(tPlot, cLowPlot, '--', 'Color', corderLowHigh(1,:), 'LineWidth', 1);
plot(tPlot, cHighPlot, '--', 'Color', corderLowHigh(2,:), 'LineWidth', 1);
axis('square');
xlabel('time (s)');
ylabel('estimated concentration');
ylim([-10,50]);
title(sprintf('naive model without distributed code, \\lambda = %0.2f, \\alpha = %0.4f', lambda, a));
set(gca, 'box','off','linewidth',2,'fontsize', 16);


figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
if nLow + nHigh < nOdor
    bkgMean = mean(fast_moving_average(cDistPlot(:,nLow+nHigh+1:end), avgSpan),2);
    bkgSD = std(fast_moving_average(cDistPlot(:,nLow+nHigh+1:end), avgSpan),0,2);
    PlotAsymmetricErrorPatch(tPlot, bkgMean, bkgSD, bkgSD, [0.5,0.5,0.5]);
end
plot(tPlot, fast_moving_average(cDistPlot(:,1:nLow), avgSpan), '-', 'Color', corderLowHigh(1,:), 'LineWidth', 2);
plot(tPlot, fast_moving_average(cDistPlot(:,nLow+1:nLow+nHigh), avgSpan), '-', 'Color', corderLowHigh(2,:), 'LineWidth', 2);
plot(tPlot, cLowPlot, '--', 'Color', corderLowHigh(1,:), 'LineWidth', 1);
plot(tPlot, cHighPlot, '--', 'Color', corderLowHigh(2,:), 'LineWidth', 1);
axis('square');
xlabel('time (s)');
ylabel('estimated concentration');
ylim([-10,50]);
title(sprintf('naive geometry with distributed code, \\lambda = %0.2f, \\alpha = %0.4f', lambda, a));
set(gca, 'box','off','linewidth',2,'fontsize', 16);

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
if nLow + nHigh < nOdor
    bkgMean = mean(fast_moving_average(cGeomPlot(:,nLow+nHigh+1:end), avgSpan),2);
    bkgSD = std(fast_moving_average(cGeomPlot(:,nLow+nHigh+1:end), avgSpan),0,2);
    PlotAsymmetricErrorPatch(tPlot, bkgMean, bkgSD, bkgSD, [0.5,0.5,0.5]);
end
plot(tPlot, fast_moving_average(cGeomPlot(:,1:nLow), avgSpan), '-', 'Color', corderLowHigh(1,:), 'LineWidth', 2);
plot(tPlot, fast_moving_average(cGeomPlot(:,nLow+1:nLow+nHigh), avgSpan), '-', 'Color', corderLowHigh(2,:), 'LineWidth', 2);
plot(tPlot, cLowPlot, '--', 'Color', corderLowHigh(1,:), 'LineWidth', 1);
plot(tPlot, cHighPlot, '--', 'Color', corderLowHigh(2,:), 'LineWidth', 1);
axis('square');
xlabel('time (s)');
ylabel('estimated concentration');
ylim([-10,50]);
title(sprintf('tuned geometry, \\lambda = %0.2f, \\alpha = %0.4f', lambda, a));
set(gca, 'box','off','linewidth',2,'fontsize', 16);



%% Compute and plot cumulative statistics after onset of low odors

tFast = 1.00;

tWindow = (0:dtPlot:tFast-dtPlot)';

nWindow = length(tWindow);

tic;


lowNaive = cNaivePlot(tPlot >= tOnLow & tPlot < tOnLow+tFast, 1:nLow);
lowDist = cDistPlot(tPlot >= tOnLow & tPlot < tOnLow+tFast, 1:nLow);
lowGeom = cGeomPlot(tPlot >= tOnLow & tPlot < tOnLow+tFast, 1:nLow);

highNaive = cNaivePlot(tPlot >= tOnLow & tPlot < tOnLow+tFast, nLow+1:nLow+nHigh);
highDist = cDistPlot(tPlot >= tOnLow & tPlot < tOnLow+tFast, nLow+1:nLow+nHigh);
highGeom = cGeomPlot(tPlot >= tOnLow & tPlot < tOnLow+tFast, nLow+1:nLow+nHigh);

lowCumMeanNaive = cumsum(lowNaive, 1) ./ (1:nWindow)';
lowCumMeanDist = cumsum(lowDist, 1) ./ (1:nWindow)';
lowCumMeanGeom = cumsum(lowGeom, 1) ./ (1:nWindow)';

highCumMeanNaive = cumsum(highNaive, 1) ./ (1:nWindow)';
highCumMeanDist = cumsum(highDist, 1) ./ (1:nWindow)';
highCumMeanGeom = cumsum(highGeom, 1) ./ (1:nWindow)';

lowCumVarNaive = cumsum(lowNaive.^2, 1) ./ (1:nWindow)' - lowCumMeanNaive.^2;
lowCumVarDist = cumsum(lowDist.^2, 1) ./ (1:nWindow)' - lowCumMeanDist.^2;
lowCumVarGeom = cumsum(lowGeom.^2, 1) ./ (1:nWindow)' - lowCumMeanGeom.^2;

highCumVarNaive = cumsum(highNaive.^2, 1) ./ (1:nWindow)' - highCumMeanNaive.^2;
highCumVarDist = cumsum(highDist.^2, 1) ./ (1:nWindow)' - highCumMeanDist.^2;
highCumVarGeom = cumsum(highGeom.^2, 1) ./ (1:nWindow)' - highCumMeanGeom.^2;

toc;


figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
set(gca, 'colororder', corderGeom);
plot(nan(1,1), nan(4,1), 'linewidth', 2);
% for ind = 1:nLow
%     plot(tWindow, lowCumMeanNaive(:,ind),  '-', 'Color', [corderGeom(1,:), 1-ind/nLow] , 'LineWidth', 2);
%     plot(tWindow, lowCumMeanDist(:,ind),  '-', 'Color', [corderGeom(2,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, lowCumMeanGeom(:,ind),  '-', 'Color', [corderGeom(3,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, ones(length(tWindow),1) .* cHighMean(:,ind),'Color',[corderGeom(4,:), 1-ind/nLow], 'LineWidth', 2);
% end
plot(tWindow, lowCumMeanNaive, 'Color', corderGeom(1,:), 'LineWidth', 0.5);
plot(tWindow, lowCumMeanDist, 'Color', corderGeom(2,:), 'LineWidth', 0.5);
plot(tWindow, lowCumMeanGeom, 'Color', corderGeom(3,:), 'LineWidth', 0.5);
plot(tWindow, ones(length(tWindow),1) .* cLowMean(:,1:nLow), 'Color', corderGeom(4,:), 'LineWidth', 0.5);
plot(tWindow, mean(lowCumMeanNaive,2), 'Color', corderGeom(1,:), 'LineWidth', 3);
plot(tWindow, mean(lowCumMeanDist,2), 'Color', corderGeom(2,:), 'LineWidth', 3);
plot(tWindow, mean(lowCumMeanGeom,2), 'Color', corderGeom(3,:), 'LineWidth', 3);
plot(tWindow, ones(length(tWindow),1) * mean(cLowMean(:,1:nLow),2), 'Color', corderGeom(4,:), 'LineWidth', 3);
axis('square');
xlabel('time since onset of low odors (s)')
ylabel('cumulative mean');
legend({'naive','distributed','geometry', 'baseline'})
title('low odors')
set(gca, 'box','off','linewidth',2,'fontsize', 16);

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
set(gca, 'colororder', corderGeom);
plot(nan(1,1), nan(3,1), 'linewidth', 2);
% for ind = 1:nLow
%     plot(tWindow, highCumMeanNaive(:,ind),  '-', 'Color', [corderGeom(1,:), 1-ind/nLow] , 'LineWidth', 2);
%     plot(tWindow, highCumMeanDist(:,ind),  '-', 'Color', [corderGeom(2,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, highCumMeanGeom(:,ind),  '-', 'Color', [corderGeom(3,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, ones(length(tWindow),1) .* cHighMean(:,nLow+ind),'Color',[corderGeom(4,:), 1-ind/nLow], 'LineWidth', 2);
% end
plot(tWindow, highCumMeanNaive, 'Color', corderGeom(1,:), 'LineWidth', 0.5);
plot(tWindow, highCumMeanDist, 'Color', corderGeom(2,:), 'LineWidth', 0.5);
plot(tWindow, highCumMeanGeom, 'Color', corderGeom(3,:), 'LineWidth', 0.5);
plot(tWindow, ones(length(tWindow),1) .* cLowMean(:,nLow+1:nLow+nHigh), 'Color', corderGeom(4,:), 'LineWidth', 0.5);
plot(tWindow, mean(highCumMeanNaive,2), 'Color', corderGeom(1,:), 'LineWidth', 3);
plot(tWindow, mean(highCumMeanDist,2), 'Color', corderGeom(2,:), 'LineWidth', 3);
plot(tWindow, mean(highCumMeanGeom,2), 'Color', corderGeom(3,:), 'LineWidth', 3);
plot(tWindow, ones(length(tWindow),1) * mean(cLowMean(:,nLow+1:nLow+nHigh),2), 'Color', corderGeom(4,:), 'LineWidth', 3);
axis('square');
xlabel('time since onset of low odors (s)')
ylabel('cumulative mean')
legend({'naive','distributed','geometry'})
title('high odors')
set(gca, 'box','off','linewidth',2,'fontsize', 16);

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
set(gca, 'colororder', corderGeom);
plot(nan(1,1), nan(3,1), 'linewidth', 2);
% for ind = 1:nLow
%     plot(tWindow, lowCumVarNaive(:,ind),  '-', 'Color', [corderGeom(1,:), 1-ind/nLow] , 'LineWidth', 2);
%     plot(tWindow, lowCumVarDist(:,ind),  '-', 'Color', [corderGeom(2,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, lowCumVarGeom(:,ind),  '-', 'Color', [corderGeom(3,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, ones(length(tWindow),1) .* cHighVar(:,ind),'Color',[corderGeom(4,:), 1-ind/nLow], 'LineWidth', 2);
% end
plot(tWindow, lowCumVarNaive, 'Color', corderGeom(1,:), 'LineWidth', 0.5);
plot(tWindow, lowCumVarDist, 'Color', corderGeom(2,:), 'LineWidth', 0.5);
plot(tWindow, lowCumVarGeom, 'Color', corderGeom(3,:), 'LineWidth', 0.5);
plot(tWindow, ones(length(tWindow),1) .* cLowVar(:,1:nLow), 'Color', corderGeom(4,:), 'LineWidth', 0.5);
plot(tWindow, mean(lowCumVarNaive,2), 'Color', corderGeom(1,:), 'LineWidth', 3);
plot(tWindow, mean(lowCumMeanDist,2), 'Color', corderGeom(2,:), 'LineWidth', 3);
plot(tWindow, mean(lowCumVarGeom,2), 'Color', corderGeom(3,:), 'LineWidth', 3);
plot(tWindow, ones(length(tWindow),1) * mean(cLowVar(:,1:nLow),2), 'Color', corderGeom(4,:), 'LineWidth', 3);
axis('square');
xlabel('time since onset of low odors (s)')
ylabel('cumulative variance')
legend({'naive','distributed','geometry'})
title('low odors')
set(gca, 'box','off','linewidth',2,'fontsize', 16);

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
set(gca, 'colororder', corderGeom);
plot(nan(1,1), nan(3,1), 'linewidth', 2);
% for ind = 1:nLow
%     plot(tWindow, highCumVarNaive(:,ind),  '-', 'Color', [corderGeom(1,:), 1-ind/nLow] , 'LineWidth', 2);
%     plot(tWindow, highCumVarDist(:,ind),  '-', 'Color', [corderGeom(2,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, highCumVarGeom(:,ind),  '-', 'Color', [corderGeom(3,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, ones(length(tWindow),1) .* cHighVar(:,ind+nLow),'Color',[corderGeom(4,:), 1-ind/nLow], 'LineWidth', 2);
% end
plot(tWindow, highCumVarNaive, 'Color', corderGeom(1,:), 'LineWidth', 0.5);
plot(tWindow, highCumVarDist, 'Color', corderGeom(2,:), 'LineWidth', 0.5);
plot(tWindow, highCumVarGeom, 'Color', corderGeom(3,:), 'LineWidth', 0.5);
plot(tWindow, ones(length(tWindow),1) .* cLowVar(:,nLow+1:nLow+nHigh), 'Color', corderGeom(4,:), 'LineWidth', 0.5);
plot(tWindow, mean(highCumVarNaive,2), 'Color', corderGeom(1,:), 'LineWidth', 3);
plot(tWindow, mean(highCumVarDist,2), 'Color', corderGeom(2,:), 'LineWidth', 3);
plot(tWindow, mean(highCumVarGeom,2), 'Color', corderGeom(3,:), 'LineWidth', 3);
plot(tWindow, ones(length(tWindow),1) * mean(cLowVar(:,nLow+1:nLow+nHigh),2), 'Color', corderGeom(4,:), 'LineWidth', 3);
axis('square');
xlabel('time since onset of low odors (s)')
ylabel('cumulative variance')
legend({'naive','distributed','geometry'})
title('high odors')
set(gca, 'box','off','linewidth',2,'fontsize', 16);

%% Compute and plot cumulative statistics after onset of high odors

tFast = 1.00;

tWindow = (0:dtPlot:tFast-dtPlot)';

nWindow = length(tWindow);

tic;

% zNaive = rInhNaive;
% zDist = rInhDist * Gnaive';
% zGeom = rInhGeom * Ggeom';

lowNaive = cNaivePlot(tPlot >= tOnHigh & tPlot < tOnHigh+tFast, 1:nLow);
lowDist = cDistPlot(tPlot >= tOnHigh & tPlot < tOnHigh+tFast, 1:nLow);
lowGeom = cGeomPlot(tPlot >= tOnHigh & tPlot < tOnHigh+tFast, 1:nLow);

highNaive = cNaivePlot(tPlot >= tOnHigh & tPlot < tOnHigh+tFast, nLow+1:nLow+nHigh);
highDist = cDistPlot(tPlot >= tOnHigh & tPlot < tOnHigh+tFast, nLow+1:nLow+nHigh);
highGeom = cGeomPlot(tPlot >= tOnHigh & tPlot < tOnHigh+tFast, nLow+1:nLow+nHigh);

lowCumMeanNaive = cumsum(lowNaive, 1) ./ (1:nWindow)';
lowCumMeanDist = cumsum(lowDist, 1) ./ (1:nWindow)';
lowCumMeanGeom = cumsum(lowGeom, 1) ./ (1:nWindow)';

highCumMeanNaive = cumsum(highNaive, 1) ./ (1:nWindow)';
highCumMeanDist = cumsum(highDist, 1) ./ (1:nWindow)';
highCumMeanGeom = cumsum(highGeom, 1) ./ (1:nWindow)';

lowCumVarNaive = cumsum(lowNaive.^2, 1) ./ (1:nWindow)' - lowCumMeanNaive.^2;
lowCumVarDist = cumsum(lowDist.^2, 1) ./ (1:nWindow)' - lowCumMeanDist.^2;
lowCumVarGeom = cumsum(lowGeom.^2, 1) ./ (1:nWindow)' - lowCumMeanGeom.^2;

highCumVarNaive = cumsum(highNaive.^2, 1) ./ (1:nWindow)' - highCumMeanNaive.^2;
highCumVarDist = cumsum(highDist.^2, 1) ./ (1:nWindow)' - highCumMeanDist.^2;
highCumVarGeom = cumsum(highGeom.^2, 1) ./ (1:nWindow)' - highCumMeanGeom.^2;

toc;


figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
set(gca, 'colororder', corderGeom);
plot(nan(1,1), nan(4,1), 'linewidth', 2);
% for ind = 1:nLow
%     plot(tWindow, lowCumMeanNaive(:,ind),  '-', 'Color', [corderGeom(1,:), 1-ind/nLow] , 'LineWidth', 2);
%     plot(tWindow, lowCumMeanDist(:,ind),  '-', 'Color', [corderGeom(2,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, lowCumMeanGeom(:,ind),  '-', 'Color', [corderGeom(3,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, ones(length(tWindow),1) .* cHighMean(:,ind),'Color',[corderGeom(4,:), 1-ind/nLow], 'LineWidth', 2);
% end
plot(tWindow, lowCumMeanNaive, 'Color', corderGeom(1,:), 'LineWidth', 0.5);
plot(tWindow, lowCumMeanDist, 'Color', corderGeom(2,:), 'LineWidth', 0.5);
plot(tWindow, lowCumMeanGeom, 'Color', corderGeom(3,:), 'LineWidth', 0.5);
plot(tWindow, ones(length(tWindow),1) .* cHighMean(:,1:nLow), 'Color', corderGeom(4,:), 'LineWidth', 0.5);
plot(tWindow, mean(lowCumMeanNaive,2), 'Color', corderGeom(1,:), 'LineWidth', 3);
plot(tWindow, mean(lowCumMeanDist,2), 'Color', corderGeom(2,:), 'LineWidth', 3);
plot(tWindow, mean(lowCumMeanGeom,2), 'Color', corderGeom(3,:), 'LineWidth', 3);
plot(tWindow, ones(length(tWindow),1) * mean(cHighMean(:,1:nLow),2), 'Color', corderGeom(4,:), 'LineWidth', 3);
axis('square');
xlabel('time since onset of high odors (s)')
ylabel('cumulative mean');
legend({'naive','distributed','geometry', 'baseline'})
title('low odors')
set(gca, 'box','off','linewidth',2,'fontsize', 16);

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
set(gca, 'colororder', corderGeom);
plot(nan(1,1), nan(3,1), 'linewidth', 2);
% for ind = 1:nLow
%     plot(tWindow, highCumMeanNaive(:,ind),  '-', 'Color', [corderGeom(1,:), 1-ind/nLow] , 'LineWidth', 2);
%     plot(tWindow, highCumMeanDist(:,ind),  '-', 'Color', [corderGeom(2,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, highCumMeanGeom(:,ind),  '-', 'Color', [corderGeom(3,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, ones(length(tWindow),1) .* cHighMean(:,nLow+ind),'Color',[corderGeom(4,:), 1-ind/nLow], 'LineWidth', 2);
% end
plot(tWindow, highCumMeanNaive, 'Color', corderGeom(1,:), 'LineWidth', 0.5);
plot(tWindow, highCumMeanDist, 'Color', corderGeom(2,:), 'LineWidth', 0.5);
plot(tWindow, highCumMeanGeom, 'Color', corderGeom(3,:), 'LineWidth', 0.5);
plot(tWindow, ones(length(tWindow),1) .* cHighMean(:,nLow+1:nLow+nHigh), 'Color', corderGeom(4,:), 'LineWidth', 0.5);
plot(tWindow, mean(highCumMeanNaive,2), 'Color', corderGeom(1,:), 'LineWidth', 3);
plot(tWindow, mean(highCumMeanDist,2), 'Color', corderGeom(2,:), 'LineWidth', 3);
plot(tWindow, mean(highCumMeanGeom,2), 'Color', corderGeom(3,:), 'LineWidth', 3);
plot(tWindow, ones(length(tWindow),1) * mean(cHighMean(:,nLow+1:nLow+nHigh),2), 'Color', corderGeom(4,:), 'LineWidth', 3);
axis('square');
xlabel('time since onset of high odors (s)')
ylabel('cumulative mean')
legend({'naive','distributed','geometry'})
title('high odors')
set(gca, 'box','off','linewidth',2,'fontsize', 16);

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
set(gca, 'colororder', corderGeom);
plot(nan(1,1), nan(3,1), 'linewidth', 2);
% for ind = 1:nLow
%     plot(tWindow, lowCumVarNaive(:,ind),  '-', 'Color', [corderGeom(1,:), 1-ind/nLow] , 'LineWidth', 2);
%     plot(tWindow, lowCumVarDist(:,ind),  '-', 'Color', [corderGeom(2,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, lowCumVarGeom(:,ind),  '-', 'Color', [corderGeom(3,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, ones(length(tWindow),1) .* cHighVar(:,ind),'Color',[corderGeom(4,:), 1-ind/nLow], 'LineWidth', 2);
% end
plot(tWindow, lowCumVarNaive, 'Color', corderGeom(1,:), 'LineWidth', 0.5);
plot(tWindow, lowCumVarDist, 'Color', corderGeom(2,:), 'LineWidth', 0.5);
plot(tWindow, lowCumVarGeom, 'Color', corderGeom(3,:), 'LineWidth', 0.5);
plot(tWindow, ones(length(tWindow),1) .* cHighVar(:,1:nLow), 'Color', corderGeom(4,:), 'LineWidth', 0.5);
plot(tWindow, mean(lowCumVarNaive,2), 'Color', corderGeom(1,:), 'LineWidth', 3);
plot(tWindow, mean(lowCumMeanDist,2), 'Color', corderGeom(2,:), 'LineWidth', 3);
plot(tWindow, mean(lowCumVarGeom,2), 'Color', corderGeom(3,:), 'LineWidth', 3);
plot(tWindow, ones(length(tWindow),1) * mean(cHighVar(:,1:nLow),2), 'Color', corderGeom(4,:), 'LineWidth', 3);
axis('square');
xlabel('time since onset of high odors (s)')
ylabel('cumulative variance')
legend({'naive','distributed','geometry'})
title('low odors')
set(gca, 'box','off','linewidth',2,'fontsize', 16);

figure('Position',[200,500,500,700],'WindowStyle','docked');
hold on;
set(gca, 'colororder', corderGeom);
plot(nan(1,1), nan(3,1), 'linewidth', 2);
% for ind = 1:nLow
%     plot(tWindow, highCumVarNaive(:,ind),  '-', 'Color', [corderGeom(1,:), 1-ind/nLow] , 'LineWidth', 2);
%     plot(tWindow, highCumVarDist(:,ind),  '-', 'Color', [corderGeom(2,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, highCumVarGeom(:,ind),  '-', 'Color', [corderGeom(3,:), 1-ind/nLow], 'LineWidth', 2);
%     plot(tWindow, ones(length(tWindow),1) .* cHighVar(:,ind+nLow),'Color',[corderGeom(4,:), 1-ind/nLow], 'LineWidth', 2);
% end
plot(tWindow, highCumVarNaive, 'Color', corderGeom(1,:), 'LineWidth', 0.5);
plot(tWindow, highCumVarDist, 'Color', corderGeom(2,:), 'LineWidth', 0.5);
plot(tWindow, highCumVarGeom, 'Color', corderGeom(3,:), 'LineWidth', 0.5);
plot(tWindow, ones(length(tWindow),1) .* cHighVar(:,nLow+1:nLow+nHigh), 'Color', corderGeom(4,:), 'LineWidth', 0.5);
plot(tWindow, mean(highCumVarNaive,2), 'Color', corderGeom(1,:), 'LineWidth', 3);
plot(tWindow, mean(highCumVarDist,2), 'Color', corderGeom(2,:), 'LineWidth', 3);
plot(tWindow, mean(highCumVarGeom,2), 'Color', corderGeom(3,:), 'LineWidth', 3);
plot(tWindow, ones(length(tWindow),1) * mean(cHighVar(:,nLow+1:nLow+nHigh),2), 'Color', corderGeom(4,:), 'LineWidth', 3);
axis('square');
xlabel('time since onset of high odors (s)')
ylabel('cumulative variance')
legend({'naive','distributed','geometry'})
title('high odors')
set(gca, 'box','off','linewidth',2,'fontsize', 16);


%%

function [ rExc, rInh, cEst ] = IntegratePoissonSamplingCircuit(sMat, nT, dt, nSens, nGranule, r0, tauE, tauI, lambda, AG, G)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

rExc = nan(nT, nSens);
rInh = nan(nT, nGranule);

rExc(1,:) = 1./r0;
rInh(1,:) = 0;

% Integrate the model using Euler-Maruyama
for indT = 2:nT

    drExc = sMat(indT, :) - rExc(indT-1,:) .* (r0 + rInh(indT-1,:) * AG');
    drInh = (rExc(indT-1,:) - 1) * AG - lambda * sign(rInh(indT-1,:) * G') * G;

    rExc(indT,:) = rExc(indT-1,:) + dt * drExc / tauE;
    rInh(indT,:) = rInh(indT-1,:) + dt * drInh / tauI + sqrt(2 * dt / tauI) * randn(1, nGranule);

end

% Compute the concentration estimate
cEst = rInh * G';

end

function [cMean, cVar] = ComputeBaselineMeanAndVarianceLangevin(dtMALA, nMALA, nBurn, nOdor, nSens, A, lambda, r0, sVec, B)

cMean = zeros(1,nOdor);
c2mom = zeros(1,nOdor);

% cCur = ones(1, nOdor) * r0;
%
% for indT = 1:nMALA
% 
%     % Evaluate the log-posterior gradient
%     drLow = (sVec ./ (r0 + cCur*A') - 1)*A - lambda * sign(cCur);
% 
% 
%     cCur = cCur + dtMALA * drLow * (B'*B) + sqrt(2 * dtMALA) * randn(1, nOdor) * B';
% 
%     if indT > nBurn
% 
%         muP = cMean;
%         vaP = c2mom;
% 
%         cMean = muP + (cCur - muP) / (indT - nBurn);
%         c2mom = vaP + (cCur - cMean) .* (cCur - muP);
%     end
% end

rExc = ones(1, nSens) * 1./r0;
rInh = zeros(1,nOdor);

for indT = 1:nMALA

    % Evaluate the log-posterior gradient
    drExc = sVec - rExc .* (r0 + rInh * A');
    drInh = (rExc - 1) * A - lambda * sign(rInh);

    rExc = rExc + dtMALA * drExc;
    rInh = rInh + dtMALA * drInh + sqrt(2 * dtMALA) * randn(1, nOdor);

    if indT > nBurn

        muP = cMean;
        vaP = c2mom;

        cMean = muP + (rInh - muP) / (indT - nBurn);
        c2mom = vaP + (rInh - cMean) .* (rInh - muP);
    end
end

cVar = c2mom / (nMALA - nBurn - 1);

end
