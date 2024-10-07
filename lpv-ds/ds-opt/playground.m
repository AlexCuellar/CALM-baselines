%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 1 - OPTION 1 (DATA LOADING): Load CORL-paper Datasets %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear all; clc
%%%%%%%%%%%%%%%%%%%%%%%%% Select a Dataset %%%%%%%%%%%%%%%%%%%%%%%%%%
% 1:  Messy Snake Dataset   (2D) *
% 2:  L-shape Dataset       (2D) *
% 3:  A-shape Dataset       (2D) * 
% 4:  S-shape Dataset       (2D) * 
% 5:  Dual-behavior Dataset (2D) *
% 6:  Via-point Dataset     (3D) * 9  trajectories recorded at 100Hz
% 7:  Sink Dataset          (3D) * 11 trajectories recorded at 100Hz
% 8:  CShape bottom         (3D) * 16 trajectories recorded at 100Hz
% 9:  CShape top            (3D) --12 trajectories recorded at 100Hz
% 10: CShape all            (3D) -- x trajectories recorded at 100Hz
% 12: test                  (2D) Alex messing around 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pkg_dir         = pwd;
chosen_dataset  = 15; 
sub_sample      = 1; % '>2' for real 3D Datasets, '1' for 2D toy datasets
nb_trajectories = 4; % Only for real 3D data
[Data, Data_sh, att, x0_all, data, dt, data_raw] = load_dataset_DS(pkg_dir, chosen_dataset, sub_sample, nb_trajectories);
% Position/Velocity Trajectories
vel_samples = 10; vel_size = 0.5; 
[h_data, h_att, h_vel] = plot_reference_trajectories_DS(Data, att, vel_samples, vel_size);

% Extract Position and Velocities
M           = size(Data,1)/2;    
Xi_ref      = Data(1:M,:);
Xi_dot_ref  = Data(M+1:end,:);   
axis_limits = axis;
disp("data!!")
data_raw{1}(1:M,:)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 1 - OPTION 2 (DATA LOADING): Load Motions from LASA Handwriting Dataset %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% UNCOMMENT BLOCK IF YOU WANT TO USE DATA FROM LASA HANDWRITING DATASET
% % Choose DS LASA Dataset to load
% clear all; close all; clc
% 
% % Select one of the motions from the LASA Handwriting Dataset
% sub_sample      = 5; % Each trajectory has 1000 samples when set to '1'
% nb_trajectories = 7; % Maximum 7, will select randomly if <7
% [Data, Data_sh, att, x0_all, ~, dt] = load_LASA_dataset_DS(sub_sample, nb_trajectories);
% 
% % Position/Velocity Trajectories
% vel_samples = 15; vel_size = 0.5; 
% [h_data, h_att, h_vel] = plot_reference_trajectories_DS(Data, att, vel_samples, vel_size);
% 
% % Extract Position and Velocities
% M          = size(Data,1)/2;    
% Xi_ref     = Data(1:M,:);
% Xi_dot_ref = Data(M+1:end,:);  

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Step 2 (GMM FITTING): Fit GMM to Trajectory Data %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% GMM Estimation Algorithm %%%%%%%%%%%%%%%%%%%%%%
% 0: Physically-Consistent Non-Parametric (Collapsed Gibbs Sampler)
% 1: GMM-EM Model Selection via BIC
% 2: CRP-GMM (Collapsed Gibbs Sampler)
est_options = [];
est_options.type             = 0;   % GMM Estimation Algorithm Type
% PC-GMM IS 0 BUT LIGHT_SPEED SHOULD BE COMPILED
% est_options.type             = 1;   % GMM Estimation Algorithm Type 

% If algo 1 selected:
est_options.maxK             = 10;  % Maximum Gaussians for Type 1
est_options.fixed_K          = [];  % Fix K and estimate with EM for Type 1

% If algo 0 or 2 selected:
est_options.samplerIter      = 50;  % Maximum Sampler Iterations
                                    % For type 0: 20-50 iter is sufficient
                                    % For type 2: >100 iter are needed
                                    
est_options.do_plots         = 1;   % Plot Estimation Statistics
% Size of sub-sampling of trajectories
% 1/2 for 2D datasets, >2/3 for real
nb_data = length(Data);
sub_sample = 1;
if nb_data > 500
    sub_sample = 2;
elseif nb_data > 1000
        sub_sample = 3;
end
est_options.sub_sample       = sub_sample;       

% Metric Hyper-parameters
est_options.estimate_l       = 1;   % '0/1' Estimate the lengthscale, if set to 1
est_options.l_sensitivity    = 2;   % lengthscale sensitivity [1-10->>100]
                                    % Default value is set to '2' as in the
                                    % paper, for very messy, close to
                                    % self-intersecting trajectories, we
                                    % recommend a higher value
est_options.length_scale     = [];  % if estimate_l=0 you can define your own
                                    % l, when setting l=0 only
                                    % directionality is taken into account

% Fit GMM to Trajectory Data
[Priors, Mu, Sigma] = fit_gmm(Xi_ref, Xi_dot_ref, est_options);

%% Generate GMM data structure for DS learning

%%%% This re-ordering needed to linearize the linear DS @ attractor
% Order Gaussian parameters based on closeness to attractor 
[idx] = knnsearch(Mu', att', 'k', size(Mu,2));
Priors = Priors(:,idx);
Mu     = Mu(:,idx);
Sigma  = Sigma(:,:,idx);

% Make the closest Gaussian isotropic and place it at the attractor location
Sigma(:,:,1) = 1.*max(diag(Sigma(:,:,1)))*eye(M);
Mu(:,1) = att;

clear ds_gmm; ds_gmm.Mu = Mu; ds_gmm.Sigma = Sigma; ds_gmm.Priors = Priors; 

% (Recommended!) Step 2.1: Dilate the Covariance matrices that are too thin
% This is recommended to get smoother streamlines/global dynamics
adjusts_C  = 1;
if adjusts_C  == 1 
    if M == 2
        tot_dilation_factor = 1; rel_dilation_fact = 0.25;
    elseif M == 3
        tot_dilation_factor = 1; rel_dilation_fact = 0.75;        
    end
    Sigma_ = adjust_Covariances(ds_gmm.Priors, ds_gmm.Sigma, tot_dilation_factor, rel_dilation_fact);
    ds_gmm.Sigma = Sigma_;
end   

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%  Step 3 (DS ESTIMATION): ESTIMATE SYSTEM DYNAMICS MATRICES  %%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% DS OPTIMIZATION OPTIONS %%%%%%%%%%%%%%%%%%%%%% 
% Type of constraints/optimization 
lyap_constr = 2;      % 0:'convex':     A' + A < 0 (Proposed in paper)
                      % 2:'non-convex': A'P + PA < -Q given P (Proposed in paper)                                 
init_cvx    = 1;      % 0/1: initialize non-cvx problem with cvx 
symm_constr = 0;      % This forces all A's to be symmetric (good for simple reaching motions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P_opt = eye(M);
[A_k, b_k, ~] = optimize_lpv_ds_from_data(Data, att, lyap_constr, ds_gmm, P_opt, init_cvx, symm_constr);
A_k
b_k
ds_lpv = @(x) lpv_ds(x, ds_gmm, A_k, b_k);
ds_lpv
opt_sim = [];
opt_sim.dt    = 0.005;   
opt_sim.i_max = 10000;
opt_sim.tol   = 0.005;
opt_sim.plot  = 0;
opt_sim.perturbation.type = 'rdp';
opt_sim.perturbation.t0 = 0.0;
opt_sim.perturbation.tf = 1.0;
opt_sim.perturbation.dx = [.5; 0];
% [x_sim, ~]    = simulate_simple(x0_all ,[],ds_lpv, opt_sim);
[x_sim, ~]    = Simulation(x0_all ,[],ds_lpv, opt_sim);
x_sim
delete penm_log.txt
%% %%%%%%%%%%%%    Plot Resulting DS  %%%%%%%%%%%%%%%%%%%
% % Fill in plotting options
ds_plot_options = [];
ds_plot_options.sim_traj  = 1;            % To simulate trajectories from x0_all
ds_plot_options.x0_all    = x0_all;       % Intial Points
x0_all
ds_plot_options.init_type = 'ellipsoid';  % For 3D DS, to initialize streamlines
                                          % 'ellipsoid' or 'cube'  
ds_plot_options.nb_points = 30;           % No of streamlines to plot (3D)
ds_plot_options.plot_vol  = 1;            % Plot volume of initial points (3D)
ds_plot_options.limits    = axis_limits;
[hd, hr, x_sim] = visualize_simple(Xi_ref, x_sim, ds_plot_options);

limits = axis;
switch lyap_constr
    case 0
        title('GMM-based LPV-DS with QLF', 'Interpreter','LaTex','FontSize',20)
    case 1
        title('GMM-based LPV-DS with P-QLF (v0) ', 'Interpreter','LaTex','FontSize',20)
    case 2
        title('GMM-based LPV-DS with P-QLF', 'Interpreter','LaTex','FontSize',20)
end

if ds_plot_options.sim_traj
    nb_traj       = size(x_sim,3);
    dtwd = zeros(1,nb_traj);
    for n=1:nb_traj
        dtwd(1,n) = dtw(x_sim(:,:,n)',data_raw{n}(1:M,:)',20)
    end
    fprintf('LPV-DS got DTWD of reproduced trajectories: %2.4f +/- %2.4f \n', mean(dtwd),std(dtwd));
end
