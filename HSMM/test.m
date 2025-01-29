function [] = alex_test(dataset_name)
close all;
addpath('./m_fcts/');

demos = [];
load('alex_data/' + dataset_name + '.mat')
nbSamples = size(demos,1); %Number of demonstration
% load('data/2Dletters/S.mat')
nbData = 0;
for i = 1:nbSamples
    demos{i}.pos = [demos{i}.pos repmat(demos{i}.pos(:,end),1,100)];
    demos{i}.vel = [demos{i}.vel repmat(demos{i}.vel(:,end),1,100)];
    demos{i}.acc = [demos{i}.acc repmat(demos{i}.acc(:,end),1,100)];
    nbData = nbData + size(demos{i}.pos,2);
end
nbData = round(nbData/nbSamples)
epsilon = .1;
use_perturbation = false;
perturbation.t0 = 3;
perturbation.tf = 3.8;
perturbation.x_final = [4.5; 0];
%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbD = 20; %Time window for LQR computation
onlyDur = 0; %Forward variable parameter (0 for standard HSMM computation, 1 for HSMM considering only duration)

model.nbStates = 30; %Number of states
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 0.02; %Time step duration
model.minSigmaPd = 1E-3; %Minimum variance of state duration (regularization term)
model.rfactor = 1E-5;	%Control cost in LQR (to be set carefully because infinite horizon LQR can suffer mumerical instability)

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;

%Artificial trigerring of external input
u = zeros(nbData,1); %No perturbation signal
% u = [zeros(20,1); ones(20,1); zeros(60,1)]; %Simulation of perturbation signal


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Integration with higher order Taylor series expansion
A1d = zeros(model.nbDeriv);
for i=0:model.nbDeriv-1
	A1d = A1d + diag(ones(model.nbDeriv-i,1),i) * model.dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(model.nbDeriv,1); 
for i=1:model.nbDeriv
	B1d(model.nbDeriv-i+1) = model.dt^i * 1/factorial(i); %Discrete 1D
end
A0 = kron(A1d, eye(model.nbVarPos)); %Discrete nD
B0 = kron(B1d, eye(model.nbVarPos)); %Discrete nD

A = [A0, zeros(model.nbVar,1); zeros(1,model.nbVar), 1]; %Augmented A
B = [B0; zeros(1,model.nbVarPos)]; %Augmented B


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data = [];
for n=1:nbSamples
	s(n).Data=[];
	for m=1:model.nbDeriv
		if m==1
			dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
		else
			dTmp = gradient(dTmp) / model.dt; %Compute derivatives
		end
		s(n).Data = [s(n).Data; dTmp];
	end
	Data0(:,n) = s(n).Data(:,1);
    DataF(:,n) = s(n).Data(:,end);
	Data = [Data s(n).Data]; 
end
% Data0 = mean(Data0,2);
DataF = mean(Data0,2);

inits = Data0; % Initialize at each first state of demos
% inits = [5.3; 3; 0; 0]; % Bespoke initialization

%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Learning');
%model = init_GMM_kmeans(Data, model);
model = init_GMM_kbins(Data, model, nbSamples);

% %Random initialization
% model.Trans = rand(model.nbStates,model.nbStates);
% model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);
% model.StatesPriors = rand(model.nbStates,1);
% model.StatesPriors = model.StatesPriors/sum(model.StatesPriors);

%Left-right model initialization
model.Trans = zeros(model.nbStates);
for i=1:model.nbStates-1
	model.Trans(i,i) = 1-(model.nbStates/nbData);
	model.Trans(i,i+1) = model.nbStates/nbData;
end
model.Trans(model.nbStates,model.nbStates) = 1.0;
model.StatesPriors = zeros(model.nbStates,1);
model.StatesPriors(1) = 1;
model.Priors = ones(model.nbStates,1);

%EM parameters learning
model.params_diagRegFact = 1E-3;
[model,H] = EM_HMM(s, model);

%Removal of self-transition (for HSMM representation) and normalization
model.Trans = model.Trans - diag(diag(model.Trans)) + eye(model.nbStates)*realmin;
model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);

%Post-estimation of the state duration from data 
for i=1:model.nbStates
	st(i).d=[];
end
[~,hmax] = max(H);
currState = hmax(1);
cnt = 1;
for t=1:length(hmax)
	if (hmax(t)==currState)
		cnt = cnt+1;
	else
		st(currState).d = [st(currState).d log(cnt)];
		cnt = 1;
		currState = hmax(t);
	end
end
st(currState).d = [st(currState).d log(cnt)];

%Set state duration manually (as an example) with: u=0 -> normal duration, u=1 -> Twice longer duration
for i=1:model.nbStates
	model.gmm_Pd(i).nbStates = 2; %Two Gaussians composing the state duration probability
	model.gmm_Pd(i).Priors = ones(model.gmm_Pd(i).nbStates,1);
	%First Gaussian: normal behavior
	model.gmm_Pd(i).Mu(:,1) = [0; mean(st(i).d)]; 
	model.gmm_Pd(i).Sigma(:,:,1) = diag([1E-2, cov(st(i).d)+model.minSigmaPd]);
	%Second Gaussian: Slow down the movement by a factor 2 if the input u is 1 
	model.gmm_Pd(i).Mu(:,2) = [1; log(exp(mean(st(i).d))*2)]; 
	model.gmm_Pd(i).Sigma(:,:,2) = diag([1E-2, cov(st(i).d)+model.minSigmaPd]);
end

%Transform model to the corresponding version with augmented covariance
model0 = model;
model.Mu = zeros(model.nbVar+1, model.nbStates);
model.Sigma = zeros(model.nbVar+1, model.nbVar+1, model.nbStates);
for i=1:model.nbStates
	model.Sigma(:,:,i) = [model0.Sigma(:,:,i)+model0.Mu(:,i)*model0.Mu(:,i)', model0.Mu(:,i); model0.Mu(:,i)', 1];
end


%% Reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
reproductions = cell(size(inits,2),1);

dx_step = zeros(2,1);

for sample = 1:size(inits,2)
    fprintf('Reproduction');
    nbPd = round(3 * nbData/model.nbStates); %Number of maximum duration step to consider in the HSMM (3 is a safety factor)
    %Initialization
    r(1).Data = [];	%Reproduction data
    h = zeros(model.nbStates,nbData); %Activation weights
    qList = zeros(nbData,1); %component id list
    c = zeros(nbData,1); %scaling factor to avoid numerical issues
    c(1) = 1; %Initialization of scaling factor
    X = [inits(:,sample); 1];	%Initial state vector
    r(1).Data = [];	%Reproduction data
    started_perturbation = false;
    
    for t=1:nbData
	    if mod(t,10)==1
		    fprintf('.');
	    end
	    r(1).Data = [r(1).Data X(1:end-1)]; %Log position
        if norm(X(1:end-1) - DataF) < epsilon
            disp("ARRIVED")
            break
        end
	    for i=1:model.nbStates
		    % Conditional Gaussian distribution given the external input "u"
		    [model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i)] = GMR(model.gmm_Pd(i), u(t), 1, 2);
		    % Pre-computation of duration probabilities
		    model.Pd(i,:) = gaussPDF(log(1:nbPd), model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i)) + realmin;
		    % The rescaling formula below can be used to guarantee that the cumulated sum is one (to avoid numerical issues)
		    model.Pd(i,:) = model.Pd(i,:) / sum(model.Pd(i,:));
    
		    % HSMM forward variable
		    if t <= nbPd
			    if(onlyDur)
				    oTmp = 1; %Observation probability for "duration-only HSMM"
			    else
				    oTmp = prod(c(1:t) .* gaussPDF(r(1).Data(:,1:t), model0.Mu(:,i), model0.Sigma(:,:,i))'); %Observation probability for standard HSMM
				    %oTmp = prod(c(1:t) .* gaussPDF(r(1).Data(1:model.nbVarPos,1:t), model.Mu(1:model.nbVarPos,i), model.Sigma(1:model.nbVarPos,1:model.nbVarPos,i))); %Observation probability for standard HSMM
			    end
			    h(i,t) = model.StatesPriors(i) * model.Pd(i,t) * oTmp;
		    end
		    for d=1:min(t-1,nbPd)
			    if(onlyDur)
				    oTmp = 1; %Observation probability for "duration-only HSMM"
			    else
				    oTmp = prod(c(t-d+1:t) .* gaussPDF(r(1).Data(:,t-d+1:t), model0.Mu(:,i), model0.Sigma(:,:,i))'); %Observation prob. for HSMM
				    %oTmp = prod(c(t-d+1:t) .* gaussPDF(r(1).Data(1:model.nbVarPos,t-d+1:t), model.Mu(1:model.nbVarPos,i), model.Sigma(1:model.nbVarPos,1:model.nbVarPos,i))); %Observation prob. for HSMM
			    end
			    h(i,t) = h(i,t) + h(:,t-d)' * model.Trans(:,i) * model.Pd(i,d) * oTmp;
		    end
	    end
	    c(t+1) = 1/sum(h(:,t)+realmin); %Update of scaling factor
	    
	    % Predict future weights (not influenced by position data)
	    for s=t+1:t+nbD
		    h(:,s) = zeros(model.nbStates,1);
		    for i=1:model.nbStates
			    for d=1:min(s-1,nbD)
				    h(i,s) = h(i,s) + h(:,s-d)' * model.Trans(:,i) * model.Pd(i,d);
			    end
		    end
	    end
    
	    % Linear quadratic tracking
	    [~,q] = max(h(:,t:t+nbD-1),[],1); %works also for nbStates=1
	    % Riccati equation
	    P = zeros(model.nbVar+1,model.nbVar+1,nbD);
	    P(:,:,end) = inv(model.Sigma(:,:,q(end)));
	    for s=nbD-1:-1:1
		    Q = inv(model.Sigma(:,:,q(s)));
		    P(:,:,s) = Q - A' * (P(:,:,s+1) * B / (B' * P(:,:,s+1) * B + R) * B' * P(:,:,s+1) - P(:,:,s+1)) * A;
	    end
	    K = (B' * P(:,:,1) * B + R) \ B' * P(:,:,1) * A; %FB gain
	    DDX = -K * X; %Acceleration command with FB terms on augmented state (resulting in FB and FF terms)
	    
	    % Emulating that the system stays at the same position when perturbed (if the external input u is equal to 1)
	    if u(t)~=1
		    X = A*X + B*DDX; %Update position
	    end
        if use_perturbation && t >= round(perturbation.t0/model.dt)+1 && t <= round(perturbation.tf/model.dt)
           if ~started_perturbation
                dx_full = perturbation.x_final - X(1:2);
                dx_step = dx_full/(perturbation.tf - perturbation.t0);
            end
            started_perturbation = true;
            X = [r(1).Data(1:2,end) + model.dt*dx_step ; dx_step ; X(end)];
        end
	    r(1).ddx(:,t) = DDX; %Log acceleration computed from LQR
	    r(1).Pd(:,:,t) = model.Pd; %Log temporary state duration probabilities
	    qList(t) = q(1); %Log state id
    end
    h = h ./ repmat(sum(h,1),model.nbStates,1);
    fprintf('\n');
    
    reproductions{sample} = transpose(r(1).Data(1:2,:));
end

hold on;
for i = 1:size(inits,2)
    plot(reproductions{i}(:,1),reproductions{i}(:,2),"ro-",'MarkerSize',3)
end

for i = 1:nbSamples
    plot(demos{i}.pos(1,:)',demos{i}.pos(2,:)',"ko-",'MarkerSize',3)
end
hold off



