function [x_sim_all,other_sim_all,data_raw] = test_lpv(data,sim_x0,opt_sim)

[Data, Data_sh, att, x0_all, dt, data] = processDataStructure(data)

M          = size(Data,1)/2;    
Xi_ref     = Data(1:M,:);
Xi_dot_ref = Data(M+1:end,:);  
axis_limits = axis;

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

[x_sim, xd_sim]    = Simulation(x0_all ,[],ds_seds, opt_sim);
x_sim_all = [x_sim; xd_sim]
if size(sim_x0, 1) > 0
    [other_sim, otherd_sim]    = Simulation(sim_x0 ,[],ds_seds, opt_sim);
    other_sim_all = [other_sim; otherd_sim]
else
    other_sim_all = []
end


% %% %%%%%%%%%%%%    Plot Resulting DS  %%%%%%%%%%%%%%%%%%%
% % Fill in plotting options
ds_plot_options = [];
ds_plot_options.sim_traj  = 1;            % To simulate trajectories from x0_all
ds_plot_options.x0_all    = x0_all;       % Intial Points
ds_plot_options.init_type = 'ellipsoid';  % For 3D DS, to initialize streamlines
                                          % 'ellipsoid' or 'cube'  
ds_plot_options.nb_points = 30;           % No of streamlines to plot (3D)
ds_plot_options.plot_vol  = 1;            % Plot volume of initial points (3D)
ds_plot_options.limits    = axis_limits;
[hd, hr, x_sim] = visualize_simple(Xi_ref, x_sim, ds_plot_options);
title("SEDS-DS on Initial Points", 'Interpreter','LaTex','FontSize',20)
if size(sim_x0, 1) > 0
    [hd, hr, other_sim] = visualize_simple(Xi_ref, other_sim, ds_plot_options);
    title("SEDS-DS on Provided Points", 'Interpreter','LaTex','FontSize',20)
end
