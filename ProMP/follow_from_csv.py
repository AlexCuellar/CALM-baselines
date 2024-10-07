import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import os

from movement_primitives.movement_primitives.data import generate_1d_trajectory_distribution
from movement_primitives.movement_primitives.promp import ProMP


def conditional_prmomp(T_all, Y, ndim, init_pts, T_follow, perturb, dt):
    y_conditional_cov = np.array([0.025])
    promp = ProMP(n_dims=ndim, n_weights_per_dim=50)
    promp.imitate(T_all, Y)

    num_pts_predict = int(sum([T.shape[0] for T in T_all])/len(T_all))
    T_pred = np.linspace(0, num_pts_predict*dt, num_pts_predict)
    T_follow = T_pred[:]
    # print(T_follow)
    Y_mean = promp.mean_trajectory(T_follow)
    # print(Y_mean.shape)
    Y_conf = 1.96 * np.sqrt(promp.var_trajectory(T_follow))

    fig, ax = plt.subplots() # plt.figure(figsize=(10, 5))

    # ax.set_title("Training set and ProMP")
    # ax.plot(Y_mean[:, 0], Y_mean[:, 1], c="r", lw=2, label="ProMP")
    # for i in range(len(Y)):
    #     ax.plot(Y[i][:, 0], Y[i][:, 1], c="k", alpha=0.1)
    # ax.legend(loc="best")
    Y_mean = controller(Y_mean, Y_mean[0,:], perturb, dt)
    print(Y_mean.shape)
    ax.set_title("Conditional ProMPs")
    # ax.plot(Y_mean[:, 0], Y_mean[:, 1], c="k", lw=2)
    
    cond_trajs = []
    for i in range(len(Y)):
        ax.plot(Y[i][:, 0], Y[i][:, 1], c="k")
    for y_cond in init_pts:
        # print("y_cond: ", y_cond)
        cpromp = promp.condition_position(np.array([y_cond[i] for i in range(ndim)]), y_cov=y_conditional_cov, t=0., t_max=1.0)
        Y_cmean = cpromp.mean_trajectory(T_follow)
        print(Y_cmean.shape)
        Y_cconf = 1.96 * np.sqrt(cpromp.var_trajectory(T_follow))
        # print(Y_cmean)
        Y_cmean = controller(Y_cmean, y_cond, perturb, dt)
        print(Y_cmean)
        cond_trajs.append(Y_cmean)
        # print(Y_cmean)
        # ax.scatter([0], [y_cond], marker="*", s=100, label="$y_0 = %.2f$" % y_cond)
        ax.plot(Y_cmean[:, 0], Y_cmean[:, 1], c="r")
        # ax.plot(Y_mean[:, 0], Y_mean[:, 1], c="b", lw=2)
        ax.legend(loc="best")
    
    # print("NUM COND TRAJS: ", len(cond_trajs))
    plt.tight_layout()
    plt.show()
    # controller(Y_mean)
    return Y_mean, cond_trajs

def controller(traj, initial, perturb, dt):
    traj_control = [initial]
    end_point = traj[-1, :]
    t = 0
    t_align = 0
    epsilon = 5*np.linalg.norm(traj[-1, :] - traj[-2, :])
    already_perturbed = False
    perturb_step = []
    while np.linalg.norm(traj_control[-1] - end_point) > epsilon:
        # print("t: ", t, " OG: ", traj[t_align], " controlled: ", traj_control[t_align])
        direction = (traj[t_align+1, :] - traj_control[-1])/np.linalg.norm(traj[t_align+1, :] - traj_control[-1])
        speed = np.linalg.norm(traj[t_align, :] - traj[t_align + 1, :])
        movement = speed*direction
        if perturb != "" and t*dt > perturb["time_start"] and t*dt < perturb["time_end"]:
            if not already_perturbed:
                print("PERTURBED!!!!!!")
                perturb_step = dt*(perturb["dist"] - traj_control[-1])/(perturb["time_end"] - perturb["time_start"])
            movement = perturb_step
            already_perturbed = True        
        traj_control.append(traj_control[-1] + movement)
        # print("Pos: ", traj_control[-1], " Direction: ", direction, " Speed: ", speed)
        t = min(t_align + 1, traj.shape[0] - 1)
        t_align = min(t, traj.shape[0] - 2)
    return np.array(traj_control)
    

def run_dmp(dataset_name, dt, init_pts, save, perturb):
    file_names = os.listdir("data/" + dataset_name)
    print(file_names)
    
    trajs = []
    T_all = []
    num_pts_follow = 100

    for name in file_names:
        traj_temp = np.genfromtxt("data/" + dataset_name + "/" + name, delimiter=",")[:,:2]
        # if traj_temp[0, 0] > 4: 
        #     traj_temp = traj_temp + np.array([0, 1])
        num_pts = traj_temp.shape[0]
        trajs.append(traj_temp)
        T_all.append(np.linspace(0, num_pts*dt, num_pts))
        
    # num_pts_predict = int(sum([traj.shape[0] for traj in trajs])/len(trajs))
    # print(num_pts_predict)
    # for traj in trajs:
    #     T_all.append(np.linspace(0, num_pts_predict*dt, num_pts_predict))

    initial_pts = []
    try:
        initial_pts_string = init_pts[0].split(" ")
        for pt in initial_pts_string:
            dims = [float(s) for s in pt.split(",")]
            initial_pts.append(np.array(dims))
    except:
        print("using first point of each demo")
        for traj in trajs:
            initial_pts.append(traj[0, :])

    if perturb != "":
        perturb = {}
        perturb_string = args.perturb.split(" ")
        for ptb in perturb_string:
            dims = [float(s) for s in ptb.split(",")]
            perturb["time_start"] = dims[-2]
            perturb["time_end"] = dims[-1]
            perturb["dist"] = np.array(dims[:-2])
    # print(initial_pts)
    T_follow = np.linspace(0, num_pts_follow*dt, num_pts_follow)
    mean, cond_means = conditional_prmomp(T_all, trajs, 2, initial_pts, T_follow, perturb, dt)
    print(cond_means[0][0])
    # print("HERE!!!!!!!", len(cond_means))
    if(save):
        # print(dataset_name)
        dir_name = 'results/' + dataset_name + "/"
        file_name = dataset_name.split("/")[-1] # THIS IS FOR LASA SUB-FOLDER
        # print(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        csv_name = dir_name + file_name + ".csv"
        with open(csv_name, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=",")
            for i in range(mean.shape[0]):
                writer.writerow(mean[i,:].tolist())
        # print(dir_name, len(initial_pts))
        for i in range(len(initial_pts)):
            csv_name = dir_name + file_name + "_" + str(i) + ".csv"
            with open(csv_name, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=",")
                for j in range(cond_means[i].shape[0]):
                    writer.writerow(cond_means[i][j].tolist())
            csv_name_gt = dir_name + file_name + "_" + str(i) + "_gt.csv"
            with open(csv_name_gt, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=",")
                for j in range(trajs[i].shape[0]):
                    writer.writerow(trajs[i][j].tolist())


if __name__ == "__main__":
    ndim = 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_file", "-d", type=str,required=True)
    parser.add_argument("--dt", "-t", type=float, default=.02)
    parser.add_argument("--init_pts", "-i", type=str, nargs='+',required=False)
    parser.add_argument("--save", "-s", type=bool, default=False)
    parser.add_argument("--perturb", "-p", type=str, default="")

    args = parser.parse_args()
    run_dmp(args.demo_file, args.dt, args.init_pts, args.save, args.perturb)