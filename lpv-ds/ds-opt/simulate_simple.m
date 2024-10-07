function [x xd t xT x_obs]=Simulation_simple(x0,xT,fn_handle,varargin)
%
% This function simulates motion that were learnt using SEDS, which defines
% a motins as a nonlinear time-independent asymptotically stable dynamical
% systems:
%                               xd=f(x)
%
% where x is an arbitrary d dimensional variable, and xd is its first time
% derivative.
%
% The function can be called using:
%       [x xd t]=Simulation(x0,xT,Priors,Mu,Sigma)
%
% or
%       [x xd t]=Simulation(x0,xT,Priors,Mu,Sigma,options)
%
% to also send a structure of desired options.
%
% Inputs -----------------------------------------------------------------
%   o x:       d x N matrix vector representing N different starting point(s)
%   o xT:      d x 1 Column vector representing the target point
%   o fn_handle:  A handle function that only gets as input a d x N matrix,
%                 and returns the output matrix of the same dimension. Note
%                 that the output variable is the first time derivative of
%                 the input variable.
%
%   o options: A structure to set the optional parameters of the simulator.
%              The following parameters can be set in the options:
%       - .dt:      integration time step [default: 0.02]
%       - .i_max:   maximum number of iteration for simulator [default: i_max=1000]
%       - .plot     setting simulation graphic on (true) or off (false) [default: true]
%       - .tol:     A positive scalar defining the threshold to stop the
%                   simulator. If the motions velocity becomes less than
%                   tol, then simulation stops [default: 0.001]
%       - .perturbation: a structure to apply pertorbations to the robot.
%                        This variable has the following subvariables:
%       - .perturbation.type: A string defining the type of perturbations.
%                             The acceptable values are:
%                             'tdp' : target discrete perturbation
%                             'tcp' : target continuous perturbation
%                             'rdp' : robot discrete perturbation
%                             'rcp' : robot continuous perturbation
%       - .perturbation.t0:   A positive scalar defining the time when the
%                             perturbation should be applied.
%       - .perturbation.tf:   A positive scalar defining the final time for
%                             the perturbations. This variable is necessary
%                             only when the type is set to 'tcp' or 'rcp'.
%       - .perturbation.dx:   A d x 1 vector defining the perturbation's
%                             magnitude. In 'tdp' and 'rdp', it simply
%                             means a relative displacement of the
%                             target/robot with the vector dx. In 'tcp' and
%                             'rcp', the target/robot starts moving with
%                             the velocity dx.
%
% Outputs ----------------------------------------------------------------
%   o x:       d x T x N matrix containing the position of N, d dimensional
%              trajectories with the length T.
%
%   o xd:      d x T x N matrix containing the velocity of N, d dimensional
%              trajectories with the length T.
%
%   o t:       1 x N vector containing the trajectories' time.
%
%   o xT:      A matrix recording the change in the target position. Useful
%              only when 'tcp' or 'tdp' perturbations applied.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    Copyright (c) 2010 S. Mohammad Khansari-Zadeh, LASA Lab, EPFL,   %%%
%%%          CH-1015 Lausanne, Switzerland, http://lasa.epfl.ch         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The program is free for non-commercial academic use. Please contact the
% author if you are interested in using the software for commercial purposes.
% The software must not be modified or distributed without prior permission
% of the authors. Please acknowledge the authors in any academic publications
% that have made use of this code or part of it. Please use this BibTex
% reference:
% 
% S. M. Khansari-Zadeh and A. Billard, "Learning Stable Non-Linear Dynamical 
% Systems with Gaussian Mixture Models", IEEE Transaction on Robotics, 2011.
%
% To get latest upadate of the software please visit
%                          http://lasa.epfl.ch/khansari
%
% Please send your feedbacks or questions to:
%                           mohammad.khansari_at_epfl.ch

%% parsing inputs
if isempty(varargin)
    options = check_options();
else
    options = check_options(varargin{1}); % Checking the given options, and add ones are not defined.
end

d=size(x0,1); %dimension of the model
if isempty(xT)
    xT = zeros(d,1);
end

if d~=size(xT,1)
    disp('Error: length(x0) should be equal to length(xT)!')
    x=[];xd=[];t=[];
    return
end

%% setting initial values
nbSPoint=size(x0,2); %number of starting points. This enables to simulatneously run several trajectories

obs_bool = false;
obs = [];
x_obs = NaN;

%initialization
for i=1:nbSPoint
    x(:,1,i) = x0(:,i);
end
xd = zeros(size(x));
if size(xT) == size(x0)
    XT = xT;
else
    XT = repmat(xT,1,nbSPoint); %a matrix of target location (just to simplify computation)
end
            
t=0; %starting time
i=1;
while true
    %Finding xd using fn_handle.
    xd(:,i,:)=reshape(fn_handle(squeeze(x(:,i,:))-XT),[d 1 nbSPoint]);
    %%% Integration
    x(:,i+1,:)=x(:,i,:)+xd(:,i,:)*options.dt;
    t(i+1)=t(i)+options.dt;
    i=i+1;
    if i > 100
        break
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function options = check_options(varargin)
if ~isempty(varargin)
    options = varargin{1};
else
    options.dt=0.02; %to create the variable
end

if ~isfield(options,'dt') % integration time step
    options.dt = 0.02;
end
if ~isfield(options,'i_max') % maximum number of iterations
    options.i_max = 1000;
end
if ~isfield(options,'tol') % convergence tolerance
    options.tol = 0.001;
end
if ~isfield(options,'plot') % shall simulator plot the figure
    options.plot = 1;
else 
    options.plot = options.plot > 0;
end
if ~isfield(options,'perturbation') % shall simulator plot the figure
    options.perturbation.type = '';
else 
    if ~isfield(options.perturbation,'type') || ~isfield(options.perturbation,'t0') || ~isfield(options.perturbation,'dx') || ...
        ((strcmpi(options.perturbation.type,'rcp') || strcmpi(options.perturbation.type,'tcp')) && ~isfield(options.perturbation,'tf')) || ...
        (~strcmpi(options.perturbation.type,'rcp') && ~strcmpi(options.perturbation.type,'tcp') && ~strcmpi(options.perturbation.type,'rdp') && ~strcmpi(options.perturbation.type,'tdp'))
    
        disp('Invalid perturbation structure. The perturbation input is ignored!')
        options.perturbation.type = '';
    end
end