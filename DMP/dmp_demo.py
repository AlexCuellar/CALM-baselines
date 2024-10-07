from dmp_discrete import DMPs_discrete
import numpy as np
import matplotlib.pyplot as plt
import math
pi = math.pi

def PointsInCircum(r, n, x_offset, y_offset, frac_of_circle):
    pts = np.zeros([n+1, 2])
    for x in range(n+1):
        pts[x, 0] = math.cos(frac_of_circle*2*pi/n*x)*r + x_offset
        pts[x, 1] = math.sin(frac_of_circle*2*pi/n*x)*r + y_offset
    return pts

# test imitation of path run
plt.figure(figsize=(6, 4))
n_bfs = [30, 50, 100]

# a straight line to target
ref = PointsInCircum(1.5, 180, 2.5, 2.5, .25)
print(ref)
ref_dim1 = ref[:,0]
ref_dim2 = ref[:,1]

for ii, bfs in enumerate(n_bfs):
    dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs)

    dmp.imitate_path(y_des=np.array([ref_dim1, ref_dim2]))
    # change the scale of the movement
    dmp.goal[0] = ref_dim1[-1]
    dmp.goal[1] = ref_dim2[-1]

    y_track, dy_track, ddy_track = dmp.rollout(y0=[ref_dim1[0] - .4, ref_dim2[0] + .2])
    # print(y_track)
    plt.scatter(y_track[:, 0], y_track[:, 1])

print(ref_dim1 / ref_dim1[-1] * dmp.goal[0])
print(ref_dim2 / ref_dim2[-1] * dmp.goal[1])
a = plt.scatter(ref_dim1 / ref_dim1[-1] * dmp.goal[0], ref_dim2 / ref_dim2[-1] * dmp.goal[1])
plt.title('DMP imitate path')
plt.xlabel('time (ms)')
plt.ylabel('system trajectory')

plt.show()