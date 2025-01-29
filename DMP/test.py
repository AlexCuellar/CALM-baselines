from dmp_discrete import DMPs_discrete
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import argparse

pi = math.pi

def test(ndim, mean, initial_pts, perturb, dt):
    # test imitation of path run
    bfs = 100
    # print(mean)
    mean_by_dim = [mean[:,i] for i in range(ndim)]

    dmp = DMPs_discrete(n_dmps=ndim, n_bfs=bfs, dt=dt)

    dmp.imitate_path(y_des=np.array(mean_by_dim))
    # change the scale of the movement
    for d in range(ndim):
        dmp.goal[d] = mean_by_dim[d][-1]

    y_tracks = []
    for x0 in initial_pts:
        y_track = dmp.rollout(y0=x0, tau=1/(dt*mean.shape[0]), perturb=perturb)
        # print(y_track)
        for i in range(y_track.shape[0]): print(i, y_track[i,:])
        y_tracks.append(y_track)
    return y_tracks


def run_dmp(dataset_name, dt, init_pts, save, perturb_arg):
    mean = np.genfromtxt("data/" + dataset_name + ".csv", delimiter=",")[:,:2]
    print(mean.shape)
    initial_pts = []
    try:
        initial_pts_string = init_pts[0].split(" ")
        for pt in initial_pts_string:
            dims = [float(s) for s in pt.split(",")]
            initial_pts.append(dims)
    except:
        print("using first mean point as initial point")
        initial_pts.append([mean[0,d] for d in range(ndim)])
    
    print(perturb_arg)
    if perturb_arg != "":
        perturb = {}
        perturb_string = perturb_arg.split(" ")
        for ptb in perturb_string:
            dims = [float(s) for s in ptb.split(",")]
            perturb["time_start"] = dims[-2]
            perturb["time_end"] = dims[-1]
            perturb["dist"] = np.array(dims[:-2])
    else:
        perturb = perturb_arg

    print(perturb)
    print("mean: ", mean)
    print("initial_pts: ", initial_pts)
    qrys = test(2, mean, initial_pts, perturb, dt)
    print(qrys)
    plt.figure(figsize=(6, 4))
    for qry in qrys:
        plt.scatter(qry[:, 0], qry[:, 1], color="b")
    plt.scatter(mean[:,0], mean[:,1], color="r")
    plt.title('DMP imitate path')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.show()

    if(save):
        for i, qry in enumerate(qrys):
            print(i)
            if perturb_arg != "":
                csv_name = 'results/' + dataset_name + '_dmp_x0_' + str(initial_pts[i][0]) + '_y0_' + str(initial_pts[i][1]) + '_Xptb_' + str(perturb["dist"][0]) + "_Yptb_" + str(perturb["dist"][1]) + "_Tptb_" + str(perturb["time_start"]) + ".csv"
            else:
                csv_name = 'results/' + dataset_name + '_dmp_x0_' + str(initial_pts[i][0]) + '_y0_' + str(initial_pts[i][1]) + ".csv"
            with open(csv_name, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=",")
                for i in range(qry.shape[0]):
                    writer.writerow(qry[i,:].tolist() + qry[i,:].tolist())


if __name__ == "__main__":
    ndim = 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_file", "-d", type=str, required=True)
    parser.add_argument("--dt", "-t", type=float, default=.02)
    parser.add_argument("--init_pts", "-i", type=str, nargs='+',required=False)
    parser.add_argument("--save", "-s", type=bool, default=False)
    parser.add_argument("--perturb", "-p", type=str, default="")

    args = parser.parse_args()
    run_dmp(args.demo_file, args.dt, args.init_pts, args.save, args.perturb)
    