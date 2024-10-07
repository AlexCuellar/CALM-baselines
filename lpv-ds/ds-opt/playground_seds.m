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
chosen_dataset  = 2; 
sub_sample      = 1; % '>2' for real 3D Datasets, '1' for 2D toy datasets
nb_trajectories = 4; % Only for real 3D data
[Data, Data_sh, att, x0_all, data, dt, data_raw] = load_dataset_DS(pkg_dir, chosen_dataset, sub_sample, nb_trajectories);
Data
% Position/Velocity Trajectories
vel_samples = 10; vel_size = 0.5; 
[h_data, h_att, h_vel] = plot_reference_trajectories_DS(Data, att, vel_samples, vel_size);

% Extract Position and Velocities
M           = size(Data,1)/2;    
Xi_ref      = Data(1:M,:);
Xi_dot_ref  = Data(M+1:end,:);   
axis_limits = axis;
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
% 'seds-init': follows the initialization given in the SEDS code
% 0: Manually set the # of Gaussians
% 1: Do Model Selection with BIC
do_ms_bic = 1;

if do_ms_bic
    est_options = [];
    est_options.type        = 1;   % GMM Estimation Alorithm Type
    est_options.maxK        = 10;  % Maximum Gaussians for Type 1/2
    est_options.do_plots    = 1;   % Plot Estimation Statistics
    est_options.fixed_K     = [];   % Fix K and estimate with EM
    est_options.sub_sample  = 1;   % Size of sub-sampling of trajectories 
    
    [Priors0, Mu0, Sigma0] = fit_gmm([Xi_ref; Xi_dot_ref], [], est_options);
    nb_gaussians = length(Priors0);
else
    % Select manually the number of Gaussian components
    nb_gaussians = 4;
end

% Finding an initial guess for GMM's parameter
[Priors0, Mu0, Sigma0] = initialize_SEDS([Xi_ref; Xi_dot_ref],nb_gaussians);


%%  Visualize Gaussian Components and labels on clustered trajectories
% Plot Initial Estimate 
[~, est_labels] =  my_gmm_cluster([Xi_ref; Xi_dot_ref], Priors0, Mu0, Sigma0, 'hard', []);

% Visualize Estimated Parameters
[h_gmm]  = visualizeEstimatedGMM(Xi_ref,  Priors0, Mu0(1:M,:), Sigma0(1:M,1:M,:), est_labels, est_options);
title('GMM $\theta_{\gamma}=\{\pi_k,\mu^k,\Sigma^k\}$ Initial Estimate','Interpreter','LaTex');
  

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%  Step 3 (DS ESTIMATION): ESTIMATE SYSTEM DYNAMICS MATRICES  %%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% DS OPTIMIZATION OPTIONS %%%%%%%%%%%%%%%%%%%%%% 
clear options;
options.tol_mat_bias  = 10^-6;    % A very small positive scalar to avoid
                                  % instabilities in Gaussian kernel [default: 10^-1]                             
options.display       = 1;        % An option to control whether the algorithm
                                  % displays the output of each iterations [default: true]                            
options.tol_stopping  = 10^-6;    % A small positive scalar defining the stoppping
                                  % tolerance for the optimization solver [default: 10^-10]
options.max_iter      = 500;      % Maximum number of iteration forthe solver [default: i_max=1000]
options.objective     = 'mse';    % 'mse'/'likelihood'
sub_sample            = 1;

%running SEDS optimization solver
[Priors, Mu, Sigma]= SEDS_Solver(Priors0,Mu0,Sigma0,[Xi_ref(:,1:sub_sample:end); Xi_dot_ref(:,1:sub_sample:end)],options); 
ds_seds = @(x) GMR_SEDS(Priors,Mu,Sigma,x-repmat(att,[1 size(x,2)]),1:M,M+1:2*M);

opt_sim = [];
opt_sim.dt    = 0.005;   
opt_sim.i_max = 10000;
opt_sim.tol   = 0.005;
opt_sim.plot  = 0;
opt_sim.perturbation.type = 'rdp';
opt_sim.perturbation.t0 = 0.0;
opt_sim.perturbation.tf = 1.0;
opt_sim.perturbation.dx = [.5; 0];
[x_sim, ~]    = Simulation(x0_all ,[],ds_seds, opt_sim);

% %% %%%%%%%%%%%%    Plot Resulting DS  %%%%%%%%%%%%%%%%%%%
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

% ds_plot_options = [];
% ds_plot_options.sim_traj  = 1;            % To simulate trajectories from x0_all
% ds_plot_options.x0_all    = x0_all;       % Intial Points
% ds_plot_options.init_type = 'ellipsoid';  % For 3D DS, to initialize streamlines
%                                           % 'ellipsoid' or 'cube'  
% ds_plot_options.nb_points = 30;           % No of streamlines to plot (3D)
% ds_plot_options.plot_vol  = 1;            % Plot volume of initial points (3D)
% [hd, hs, hr, x_sim] = visualizeEstimatedDS(Data(1:M,:), ds_seds, ds_plot_options);
% limits = axis;
% switch options.objective
%     case 'mse'        
%         title('SEDS Dynamics with $J(\theta_{\gamma})$=MSE', 'Interpreter','LaTex','FontSize',20)
%     case 'likelihood'
%         title('SEDS Dynamics with $J(\theta_{\gamma})$= log-Likelihood', 'Interpreter','LaTex','FontSize',20)
% end


if ds_plot_options.sim_traj
    nb_traj       = size(x_sim,3);
    dtwd = zeros(1,nb_traj);
    for n=1:nb_traj
        dtwd(1,n) = dtw(x_sim(:,:,n)',data_raw{n}(1:M,:)',20)
    end
    fprintf('LPV-DS got DTWD of reproduced trajectories: %2.4f +/- %2.4f \n', mean(dtwd),std(dtwd));
end