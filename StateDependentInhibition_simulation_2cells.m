clear variables
close all


%% Simulation for the state dependent inhibtion effects
% Reproducing data for Arevian, Kapoor and Urban, Nature Neuroscience, 2008

%% Set neuron parameters

% Neuron time constants (in seconds)
tauE = 0.020;
tauI = 0.030;

% Number of sensors
nSens = 2;
% nSens = 100;

% Number of odors
nOdor = nSens;
% nOdor = 100;
% nOdor = 40;

% Number of granule cells
nGranule = 5*nOdor;
% nGranule = nOdor;

% Baseline activity
r0 = 10;

% Laplace prior strength
lambda = 2;
% lambda = 0;
% lambda = 0.25;

% Regularization strength
a = 0.5;
% a = 0;

%% Set simulation parameters

% Timestep (in seconds)
dt = 1e-4;

% Total simulation time (s)
tMax = 0.6;

% Stimulus onset time (s)
tStart = 0.100;
tEnd = 0.5;

% Time vector
t = (0:dt:tMax-dt)';
nT = length(t);

StartTime=round(tStart/dt);
EndTime=(round(tEnd/dt));
TestTime=EndTime;%StartTime+round(0.25/dt);

Intensities=linspace(1,400,80); % Intensity of input to measured cell
nIntensities=length(Intensities);
ECSRate=80; % Intensity of ECS

nmit1=1; % Identity of measured cell
nmit2=2; % Identity of ECS stimulated cells



nRepeat=32;

parfor indR=1:nRepeat
    
    
    %% Generate the affinity matrix
    
    A=ones(nSens);
    
    %% Generate the projection matrices
    G=randn(nSens,nGranule).*(rand(nSens,nGranule)<0.25);
    
    AG=A*G;
    
    rExcPoiss1 = nan(nT, nSens,nIntensities);
    rInhPoiss1 = nan(nT, nGranule,nIntensities);
    
    rExcPoiss2 = nan(nT, nSens,nIntensities);
    rInhPoiss2 = nan(nT, nGranule,nIntensities);
    
    rExcGauss1 = nan(nT, nSens,nIntensities);
    rInhGauss1 = nan(nT, nGranule,nIntensities);
    
    rExcGauss2 = nan(nT, nSens,nIntensities);
    rInhGauss2 = nan(nT, nGranule,nIntensities);
    
    for indL=1:nIntensities
        S1=zeros(nSens,nT);
        S2=zeros(nSens,nT);
        S1(:,1:nT)=r0;
        S2(:,1:nT)=r0;
        S1(nmit1,StartTime:EndTime)=Intensities(indL);
        S2(nmit1,StartTime:EndTime)=Intensities(indL);
        S2(nmit2,StartTime:EndTime)=ECSRate;
        S1=S1';
        S2=S2';
        
        
        %% Run the Poisson model without ECS
        
        rExcPoiss1(1,:,indL) = 1;
        rInhPoiss1(1,:,indL) = 0;
        
        tic;
        for indT = 2:nT
            
            drExc = S1(indT, :) - rExcPoiss1(indT-1,:,indL) .*(r0 + rInhPoiss1(indT-1,:,indL) * AG');
            drInh = (rExcPoiss1(indT-1,:,indL)-1) * AG - lambda * sign(rInhPoiss1(indT-1,:,indL) * G') * G;
            
            rExcPoiss1(indT,:,indL) = rExcPoiss1(indT-1,:,indL) + dt * drExc / tauE;
            rInhPoiss1(indT,:,indL) = rInhPoiss1(indT-1,:,indL) + dt * drInh / tauI;
            
        end
        
        
        MeasuredRate1(indR,:,indL)=rExcPoiss1(:,1,indL);
        MeanRate1(indL,indR)=mean(rExcPoiss1(StartTime:TestTime,nmit1,indL));
        
        fprintf('\tCompleted Poisson simulation without ECS in %f seconds.\n', toc);
        
        
        %% Run the Poisson model with ECS
        
        
        
        rExcPoiss2(1,:,indL) = 1;
        rInhPoiss2(1,:,indL) = 0;
        
        tic;
        for indT = 2:nT
            
            drExc = S2(indT, :) - rExcPoiss2(indT-1,:,indL) .* (r0 + rInhPoiss2(indT-1,:,indL) * AG');
            drInh = (rExcPoiss2(indT-1,:,indL) - 1) * AG - lambda * sign(rInhPoiss2(indT-1,:,indL) * G') * G;
            
            rExcPoiss2(indT,:,indL) = rExcPoiss2(indT-1,:,indL) + dt * drExc / tauE;
            rInhPoiss2(indT,:,indL) = rInhPoiss2(indT-1,:,indL) + dt * drInh / tauI;
            
        end
        
        
        MeasuredRate2(indR,:,indL)=rExcPoiss2(:,1,indL);
        MeanRate2(indL,indR)=mean(rExcPoiss2(StartTime:TestTime,nmit1,indL));
        
        fprintf('\tCompleted Poisson simulation with ECS in %f seconds.\n', toc);
        
        
        %% Run the Gaussian model without ECS
        
        rExcGauss1(1,:,indL) = 1;
        rInhGauss1(1,:,indL) = 0;
        
        tic;
        for indT = 2:nT
            
            drExc = - rExcGauss1(indT-1,:,indL)+ S1(indT, :) - ( + rInhGauss1(indT-1,:,indL) * AG');
            drInh = (rExcGauss1(indT-1,:,indL) ) * AG - lambda * sign(rInhGauss1(indT-1,:,indL) * G') * G;
            
            rExcGauss1(indT,:,indL) = rExcGauss1(indT-1,:,indL) + dt * drExc / tauE;
            rInhGauss1(indT,:,indL) = rInhGauss1(indT-1,:,indL) + dt * drInh / tauI;
            
        end
        
        
        MeasuredRateGauss1(indR,:,indL)=rExcGauss1(:,1,indL);
        MeanRateGauss1(indL,indR)=mean(rExcGauss1(StartTime:TestTime,nmit1,indL));
        
        fprintf('\tCompleted Gaussian simulation without ECS in %f seconds.\n', toc);
        
        
        %% Run the Gaussian model with ECS
        
        rExcGauss2(1,:,indL) = 1;
        rInhGauss2(1,:,indL) = 0;
        
        tic;
        for indT = 2:nT
            
            drExc = - rExcGauss2(indT-1,:,indL) + S2(indT, :) -  ( + rInhGauss2(indT-1,:,indL) * AG');
            drInh = (rExcGauss2(indT-1,:,indL) ) * AG - lambda * sign(rInhGauss2(indT-1,:,indL) * G') * G;
            
            rExcGauss2(indT,:,indL) = rExcGauss2(indT-1,:,indL) + dt * drExc / tauE;
            rInhGauss2(indT,:,indL) = rInhGauss2(indT-1,:,indL) + dt * drInh / tauI;
            
        end
        
        
        MeasuredRateGauss2(indR,:,indL)=rExcGauss2(:,1,indL);
        MeanRateGauss2(indL,indR)=mean(rExcGauss2(StartTime:TestTime,nmit1,indL));
        
        fprintf('\tCompleted Gaussian simulation without ECS in %f seconds.\n', toc);
        
        
    end
end
%% Plots

Inhib=mean(MeanRate2-MeanRate1,1)<0;

figure,
plot([min(mean(MeanRateGauss2,2)/max(mean(MeanRateGauss1,2))) 1] , [min(mean(MeanRateGauss2,2)/max(mean(MeanRateGauss1,2))) 1],'--k','linewidth',1)
hold on
plot(mean(MeanRate1,2)/max(mean(MeanRate1,2)),mean(MeanRate2,2)/max(mean(MeanRate1,2)),'linewidth',2,'color',[0.2 0.5 1])
plot(mean(MeanRateGauss1,2)/max(mean(MeanRateGauss1,2)),mean(MeanRateGauss2,2)/max(mean(MeanRateGauss1,2)),'linewidth',2,'color',[1 0.5 0.2])
ax=FormatAxis;
ax.XTick=[ 0 0.5 1];
ax.YTick=[0 0.5 1];
axis square

% Heat map of inhibition

HeatPlot=squeeze(mean(MeasuredRate2(Inhib,:,:))-mean(MeasuredRate1(Inhib,:,:)));

figure, imagesc(Intensities,t, HeatPlot)
%  caxis([-max(abs(heatPlot(:))) max(abs(heatPlot(:)))])
colormap('turbo')
set(gca,'YDir','normal')
colorbar
ax=FormatAxis;
ax.XTick=[0 200 400];
ax.YTick=[ 0.1 0.5];


Colorplot=[ 0.5 0.8 1 ; 0.2 0.5 1; 0 0.4 0.7 ; 0 0.1 0.2];

Sampled=[1:20:80];

figure,
hold on
for i =1:4
    plot(t,squeeze(MeasuredRate1(1,:,Sampled(i))),'linewidth',2,'color',Colorplot(i,:))
end
ax=FormatAxis;
ax.XTick=[ 0.1 0.5];
ax.YTick=[0 10 20];
