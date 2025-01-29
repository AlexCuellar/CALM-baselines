function [] = alex_test(varargin)
close all; clc
if nargin == 0
    demo_dir = uigetdir; %gets directory
else
    demo_dir = strcat("/home/alex/Downloads/CALM-baselines/data/",varargin{1});
end

demo_dir_array = split(demo_dir,"/");
dataset_name = demo_dir_array{end}

if nargin > 1
    save_dir = varargin{2};
end

ogFileNames = dir(fullfile(demo_dir,'*.csv'));
demos = cell(size(ogFileNames,1),1);
dt = .02;

for k = 1:length(ogFileNames)
  baseFileName = ogFileNames(k).name;
  fullFileName = fullfile(demo_dir, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  data = csvread(fullFileName)';
  data = data(1:2,:); % REMOVE VELOCITY INFO
  disp("SIZE OF DATA: ")
  size(data)
  if size(data, 1) == 2
      vs = (data(:,2:end) - data(:,1:end-1))/dt;
      data_full = [data(:,1:end-1); vs];
      size(data);
      size(data_full);
  else
      data_full = data;
  end
  demos{k} = data_full;
end

if nargin > 1
    dir_name = strcat("results/",save_dir)
    if not(isfolder(dir_name))
        mkdir(dir_name)
    end
    demo_sub_dir_name = strcat(dir_name,"/og")
    if not(isfolder(demo_sub_dir_name))
        mkdir(demo_sub_dir_name)
    end
    for i = 1:size(demos,1)
        name = strcat(demo_sub_dir_name,"/",dataset_name,"_",num2str(i),".csv")
        writematrix(demos{i}(:,:)',name)
    end
end

x0_sim = [-3; 1.7]
opt_sim = [];
opt_sim.dt    = 0.02;   
opt_sim.i_max = 10000;
opt_sim.tol   = 0.005;
opt_sim.plot  = 0;
% %%opt_sim.perturbation.type = 'rdp';
% opt_sim.perturbation.type = 'alex';
opt_sim.perturbation.t0 = 1.7;
opt_sim.perturbation.tf = 2.5;
opt_sim.perturbation.dx = [1; -1];
% opt_sim.perturbation.x_final = [4.5; 0];
disp("DEMOS: ")
% demos{1}

addpath(genpath(strcat(pwd)));
[lpv_x0_sim,lpv_other_sim] = test_lpv(demos',x0_sim,opt_sim);

if nargin > 2 & varargin{3} == true
    save_sim_x0 = true;
else
    save_sim_x0 = false;
end

if nargin > 2 & varargin{4} == true
    save_other_x0 = true;
else
    save_other_x0 = false;
end

save_other_x0
save_sim_x0

if nargin > 1
    if save_other_x0
        save_csv(lpv_other_sim,"lpv",save_dir,opt_sim.perturbation.t0,opt_sim.perturbation.dx(1),opt_sim.perturbation.dx(2))
        % save_csv(seds_other_sim,"seds",save_dir,opt_sim.perturbation.t0,opt_sim.perturbation.dx(1),opt_sim.perturbation.dx(2))
        % save_csv(lags_other_sim,"lags",save_dir,opt_sim.perturbation.t0,opt_sim.perturbation.dx(1),opt_sim.perturbation.dx(2))
        disp("SAVING OTHER")
    end
    if save_sim_x0
        save_csv(lpv_x0_sim,"lpv",save_dir,opt_sim.perturbation.t0,opt_sim.perturbation.dx(1),opt_sim.perturbation.dx(2))
        % save_csv(seds_x0_sim,"seds",save_dir,opt_sim.perturbation.t0,opt_sim.perturbation.dx(1),opt_sim.perturbation.dx(2))
        % save_csv(lags_x0_sim,"lags",save_dir,opt_sim.perturbation.t0,opt_sim.perturbation.dx(1),opt_sim.perturbation.dx(2))
        disp("SAVING SIM")
    end
end

end


function [] = save_csv(trajs,alg_type,save_dir,Tptb,Xptb,Yptb)
    dir_name = strcat("results/",save_dir)
    sub_dir_name = strcat(dir_name,"/",alg_type)
    if not(isfolder(sub_dir_name))
        mkdir(sub_dir_name)
    end
    for i = 1:size(trajs,3)
        name = strcat(sub_dir_name,"/",save_dir,"_",alg_type,"_x0_",num2str(trajs(1,1,i),3),"_y0_",num2str(trajs(2,1,i),3),"_Xptb_",num2str(Xptb,3),"_Yptb_",num2str(Yptb,3),"_Tptb_",num2str(Tptb,3),".csv")
        writematrix(trajs(:,:,i)',name)
    end
end
