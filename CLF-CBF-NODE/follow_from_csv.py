import time

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
from sklearn.preprocessing import MinMaxScaler
from scipy import interpolate

import pyLasaDataset as lasa
from os import listdir
from os.path import isfile, join
from numpy import genfromtxt
import numpy as np

class Demonstration:
    def __init__(self, demo, dt):
        vel = np.diff(demo, axis=1)/dt
        acc = np.diff(vel, axis=1)/dt
        self.pos = demo[:,:-2]
        self.vel = vel[:,:-1]
        self.acc = acc
        self.t = np.arange(0, self.pos.shape[1], 1) * dt
        self.t = self.t.reshape((1,self.pos.shape[1]))

class CSV_Dataset:
    def __init__(self, file, pad = True):
        self.dt = .02
        self.name = file
        onlyfiles = [file + f for f in listdir(file) if isfile(join(file, f))]
        demos_np = []
        for file in onlyfiles:
            demos_np.append(genfromtxt(file, delimiter=',').T)
            demos_np[-1].shape
        max_len = max([x.shape[1] for x in demos_np]) + 3 # the 3 is the padding that will get removed when we take velocity and acceleration
        
        self.demos = []
        for demo in demos_np:
            demo_padded = np.pad(demo, ((0, 0), (0, max_len - demo.shape[1])), 'edge')
            self.demos.append(Demonstration(demo_padded, self.dt))

def draw(data):
    # # Each Data object has attributes dt and demos (For documentation,
    # # refer original dataset repo:
    # # https://bitbucket.org/khansari/lasahandwritingdataset/src/master/Readme.txt)
    dt = data.dt
    demos = data.demos # list of 7 Demo objects, each corresponding to a
                            # repetition of the pattern

    # Each Demo object in demos list will have attributes pos, t, vel, acc
    # corresponding to the original .mat format described in
    # https://bitbucket.org/khansari/lasahandwritingdataset/src/master/Readme.txt
    demo_0 = demos[1]
    pos = demo_0.pos # np.ndarray, shape: (2,2000)
    vel = demo_0.vel # np.ndarray, shape: (2,2000)
    acc = demo_0.acc # np.ndarray, shape: (2,2000)
    t = demo_0.t # np.ndarray, shape: (1,2000)

    # demos_resized = []
    # start_t = [0, 5, 10, 15, 10, 5, 0]
    # for i, demo in enumerate(demos):
    #     new_demo = demo
    #     new_demo.pos = demo.pos[:,start_t[i]:]
    #     new_demo.vel = demo.vel[:,start_t[i]:]
    #     new_demo.acc = demo.acc[:,start_t[i]:]
    #     new_demo.t = demo.t[:,start_t[i]:]
    #     print(new_demo.pos.shape)
    #     print(new_demo.vel.shape)
    #     print(new_demo.acc.shape)
    #     print(new_demo.t.shape)

    # demos

    # To visualise the data (2D position and velocity) use the plot_model utility
    # lasa.utilities.plot_model(csv_data) # give any of the available

    # for demo in demos:
    #     plt.plot(demo.pos[0,:], demo.pos[1,:])
    # plt.show()
                                                    
    class Func(eqx.Module):
        mlp: eqx.nn.MLP

        def __init__(self, data_size, width_size, depth, *, key, **kwargs):
            super().__init__(**kwargs)
            initializer = jnn.initializers.orthogonal()
            self.mlp = eqx.nn.MLP(
                in_size=data_size,
                out_size=data_size,
                width_size=width_size,
                depth=depth,
                activation=jnn.tanh,
                key=key,
            )
            model_key = key
            key_weights = jrandom.split(model_key, depth+1)

            for i in range(depth+1):
                where = lambda m: m.layers[i].weight
                shape = self.mlp.layers[i].weight.shape
                self.mlp = eqx.tree_at(where, self.mlp, replace=initializer(key_weights[i], shape, dtype = jnp.float32))

        @eqx.filter_jit
        def __call__(self, t, y, args):

            return self.mlp(y)

    class Funcd(eqx.Module):
        mlp: eqx.nn.MLP

        def __init__(self, data_size, width_size, depth, *, key, **kwargs):
            super().__init__(**kwargs)
            initializer = jnn.initializers.orthogonal()
            self.mlp = eqx.nn.MLP(
                in_size=2*data_size,
                out_size=2*data_size,
                width_size=width_size,
                depth=depth,
                activation=jnn.tanh,
                key=key,
            )
            model_key = key
            key_weights = jrandom.split(model_key, depth+1)

            for i in range(depth+1):
                where = lambda m: m.layers[i].weight
                shape = self.mlp.layers[i].weight.shape
                self.mlp = eqx.tree_at(where, self.mlp, replace=initializer(key_weights[i], shape, dtype = jnp.float32))

        @eqx.filter_jit
        def __call__(self, t, yd, args):

            return self.mlp(yd)
            # return self.mlp(jnp.concatenate([yd, jnp.array([t])]))
            
    class NeuralODE(eqx.Module):
        func: Func

        def __init__(self, data_size, width_size, depth, *, key, **kwargs):
            super().__init__(**kwargs)
            self.func = Func(data_size, width_size, depth, key=key)

        def __call__(self, ts, y0):
            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(self.func),
                diffrax.Tsit5(),
                t0=ts[0],
                t1=ts[-1],
                dt0=ts[1] - ts[0],
                y0=y0,
                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                saveat=diffrax.SaveAt(ts=ts),
            )
            return solution.ys
        
    class NeuralODEd(eqx.Module):
        func: Funcd

        def __init__(self, data_size, width_size, depth, *, key, **kwargs):
            super().__init__(**kwargs)
            self.func = Funcd(data_size, width_size, depth, key=key)

        @eqx.filter_jit
        def __call__(self, ts, yd0):
            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(self.func),
                diffrax.Tsit5(),
                t0=ts[0],
                t1=ts[-1],
                dt0=ts[1] - ts[0],
                y0=yd0,
                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
                saveat=diffrax.SaveAt(ts=ts),
            )
            return solution.ys
        
    ndemos = len(demos)
    T = demos[0].t.shape[-1]
    pos_all = []
    vel_all = []
    for i in range(ndemos):
        pos_all.append((demos[i].pos).T)
        vel_all.append((demos[i].vel).T)
    posn = jnp.array(pos_all)
    veln = jnp.array(vel_all)
    tn =  jnp.array(t.T).reshape(T)
    ys_all_n = jnp.concatenate((posn,veln),axis=2)

    nsamples = T
    ts = t[0]/t[0, -1]
    ts_new = jnp.linspace(0, 1, nsamples)

    dim = posn.shape[2]

    traj_process = jnp.zeros((ndemos, nsamples, dim))
    vel_process = jnp.zeros((ndemos, nsamples, dim))

    traj_all_t_norm = []
    # time_all_process = jnp.zeros((traj_c, nsamples))

    seed = 1385

    key = jax.random.PRNGKey(seed)
    scale_state = 1

    key_trajs = jax.random.split(key, num=ndemos)

    for i in range(ndemos):
        key_dim = jax.random.split(key_trajs[i], num=dim)
        for j in range(dim):
            f = interpolate.interp1d(ts, posn[i, :, j])
            f_vel = interpolate.interp1d(ts, veln[i, :, j])
            # f = interpolate.interp1d(time_all[i][:, 0], traj_all[i][:, j])
            # ts_new = np.linspace(time_all[i][0, 0], time_all[i][-1, 0], nsamples)
            # time_all_process = time_all_process.at[i].set(ts_new)
            traj_new = f(ts_new)
            vel_new = f_vel(ts_new)
            traj_process = traj_process.at[i, :, j].set(scale_state*traj_new)
            vel_process = vel_process.at[i, :, j].set(scale_state*vel_new)

    ## Train_Test_split

    nTD = len(demos)
    traj_train = traj_process[1:nTD]
    vel_train = vel_process[1:nTD]

    traj_all_train = jnp.concatenate((traj_train, vel_train), axis=2)

    def dataloader(arrays, batch_size, *, key):
        dataset_size = arrays[0].shape[0]
        assert all(array.shape[0] == dataset_size for array in arrays)
        indices = jnp.arange(dataset_size)
        while True:
            perm = jrandom.permutation(key, indices)
            (key,) = jrandom.split(key, 1)
            start = 0
            end = batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                yield tuple(array[batch_perm] for array in arrays)
                start = end
                end = start + batch_size
                
    def main(
        dataset_size=nTD,
        batch_size=int(nTD/2),
        lr_strategy=(3e-3,),
        steps_strategy=(3000,),
        length_strategy=(1,),
        width_size=64,
        depth=3,
        seed=1000,
        plot=True,
        print_every=100,
        save_every=500,
    ):
        key = jrandom.PRNGKey(seed)
        data_key, model_key, loader_key = jrandom.split(key, 3)

        ys = traj_train
        ts = tn
        ys_dot = vel_train
        ys_all = traj_all_train

        _, length_size, data_size = ys.shape

        model = NeuralODE(data_size, width_size, depth, key=model_key)

        # Training loop like normal.
        #
        # Only thing to notice is that up until step 500 we train on only the first 10% of
        # each time series. This is a standard trick to avoid getting caught in a local
        # minimum.

        @eqx.filter_value_and_grad
        def grad_loss(model, ti, yi):
            y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:, 0])
            f = lambda t, z: model.func(t, z, _)
            # y_dot_pred = jax.vmap(jax.vmap(f, in_axes = (0, 0)), in_axes=(None, 0))(ti, yi)
            loss = jnp.mean((yi - y_pred) ** 2)
            # loss = jnp.mean((y_dot_pred - yi_dot)**2)
            return loss

        @eqx.filter_jit
        def make_step(ti, yi, model, opt_state):
            loss, grads = grad_loss(model, ti, yi)
            updates, opt_state = optim.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state

        time_all = 0

        for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
            decay_scheduler = optax.cosine_decay_schedule(lr, decay_steps=steps, alpha=0.95)

            optim = optax.adabelief(learning_rate=decay_scheduler)
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            _ts = ts[: int(length_size * length)]
            _ys = ys[:, : int(length_size * length)]
            # _ys_dot = ys_dot[:, : int(length_size * length)]
            _ys_all = ys_all_n[:, :int(length_size * length)]
            ## Single trajectory
            # for step in range(steps):
            #   start = time.time()
            #   loss, model, opt_state = make_step(_ts, _ys, _ys_dot, model, opt_state)
            #   end = time.time()
            #   if (step % print_every) == 0 or step == steps - 1:
            #     print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
            ## Batches
            for step, (yi,) in zip(
                range(steps), dataloader((_ys,), batch_size, key=loader_key)
            ):
                start = time.time()
                loss, model, opt_state = make_step(_ts, yi, model, opt_state)
                end = time.time()
                time_all += end - start
                if (step % print_every) == 0 or step == steps - 1:
                    print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
                # if (step % save_every) == 0 or step == steps - 1:
                #   eqx.tree_serialise_leaves(file_name, model)

        return ts, ys, model, time_all

    ts, ys, model, time_all = main()

    # for train_indx in range(len(demos)):
    #     plt.plot(posn[train_indx, :, 0], posn[train_indx, :, 1], c="dodgerblue", label="Real")
    #     plt.plot(posn[train_indx, 0, 0], posn[train_indx, 0, 1], c="saddlebrown", marker='o', markersize = '12', label="Start")
    #     plt.plot(posn[train_indx, -1, 0], posn[train_indx, -1, 1], c="black", marker='x', markersize = '12', label="Target")
    #     model_y = model(ts, posn[train_indx, 0])
    #     plt.plot(model_y[:, 0], model_y[:, 1], c="crimson", label="Model")
    # dist_indx = 260
    # plt.plot(model_y[dist_indx, 0], model_y[dist_indx, 1], c="black", marker='o', markersize = '12', label="Disturbance")
    # # plt.legend()
    # plt.tight_layout()
    # # plt.savefig("neural_ode.png")
    # plt.show()

    # file_name = "Worm.eqx"
    # eqx.tree_serialise_leaves(file_name, model)



    # ############################################## CBF_CLF #################################################

    import numpy as np
    import cvxpy as cp
    
    motions = []
    starting_pts = [posn[indx, 0] for indx in range(len(demos))]
    # starting_pts = [jnp.array([-3, 1.7])] # UNCOMMENT FOR CUSTOM INITIAL POINTS
    for indx in range(len(starting_pts)):
        ys = posn
        _, length_size, data_size = ys.shape
        model_load = model
        f = lambda t, z : model_load.func(t, z, _)
        xref = model_load(ts, ys[indx, 0, :])
        x = starting_pts[indx]
        print(x)
        r = jnp.array([10]) # Obstacle radius
        c = jnp.array([-25, 25]) # Obstacle center
        xall = jnp.expand_dims(x, axis=0)

        vopt_all = []

        dti = ts[1] - ts[0]
        dist_start, dist_end = int(1.7/dti), int(2.5/dti)
        dist_endpoint = jnp.array([4.5, 0])
        cmd_vel =jnp.array([[0, 0]])
        move_vector = jnp.array([0, 0])
        use_perturb = False # TURN TO TRUE TO HAVE A PERTURBATION
        for i in range(int(1*len(ts)-1)):

            if i==dist_start and use_perturb:
                move_vector = (x - dist_endpoint)/(dist_start - dist_end)
            x_t=np.asarray(x)

            # xref = np.asarray(ys[indx,i,:])

            t0i = 0

            tsi = jnp.array([0, dti])
            # print(ts[i], x_t)
            # tsi_d = jnp.array([ts[i], ts[i+1]])

            xref_t = np.asarray(xref[i, :])

            ## Only disturbance

            alpha_h = 10
            gamma = 0.1
            lambda_v = 0

            Q = np.eye(x_t.shape[0])
            G = 2*((x_t-xref_t).T)
            fx_t = np.asarray(f(ts[i], x_t))
            fxref_t = np.asarray(f(ts[i], xref_t))
            # h = -2*(((x_t-xref_t).T)@(fx_t - fxref_t)) + alpha_h*(-(((x_t-xref_t).T)@(x_t-xref_t)) + r**2) # CBF
            h = -2*(((x_t-xref_t).T)@(fx_t - fxref_t)) - alpha_h*((((x_t-xref_t).T)@(x_t-xref_t))) # CLF
            # h = -(alpha_h*(-(((x1-xref_t).T)@(x1-xref_t)) + r**2) + gamma)

            # h = alpha_h*(((x1-xref_t).T)@(x1-xref_t))+gamma

            # h = alpha_h*(-(((x1-xref_t).T)@(x1-xref_t)) + r**2) + gamma

            vopt = cp.Variable(x_t.shape[0])
            prob = cp.Problem(cp.Minimize(cp.quad_form(vopt, Q)  + lambda_v*cp.pos(G @ vopt - h)),
                            [G @ vopt <= h])


            prob.solve()

            f1 = lambda z : model_load.func(_, z,_) + vopt.value

            ## Switching mode

            xnext = f1(x)*dti + x
            if i > dist_start and i < dist_end and use_perturb:
                xnext = x + move_vector
            # xnext = solution.ys[-1,:]

            xall = jnp.append(xall, jnp.expand_dims(xnext, axis=0), axis=0)

            cmd_vel = jnp.append(cmd_vel, jnp.expand_dims(f1(x), axis=0), axis=0)

            x = xnext

            if(i%10==0):
                print(f"Time: {i}, Position: {x}")
            
        # from matplotlib.lines import Line2D
        # import matplotlib.pyplot as plt
        # import matplotlib
        # import numpy as np
        motions.append(xall)
        ## Worm disturbance
        xmin = -60
        xmax = 5
        ymin = -14
        ymax = 19

        f = lambda z : model.func(_, z, _)

    return motions, posn

dataset_name = "messy_snake"

data = CSV_Dataset("data/" + dataset_name + "/")
data = lasa.DataSet.Worm # UNCOMMENT TO TEST ON lasa data

import csv

motions, posn = draw(data)

for i in range(posn.shape[0]):
    plt.plot(posn[i, :, 0], posn[i, :, 1], c="black")

for motion in motions:
    plt.plot(motion[:, 0], motion[:, 1], c="red")

plt.show()