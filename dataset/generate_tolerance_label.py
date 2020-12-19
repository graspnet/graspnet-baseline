import os
import numpy as np
import time
import argparse
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--pos_ratio_thresh', type=float, default=0.8, help='Threshold of positive neighbor ratio[default: 0.8]')
parser.add_argument('--mu_thresh', type=float, default=0.55, help='Threshold of friction coefficient[default: 0.55]')
parser.add_argument('--num_workers', type=int, default=50, help='Worker numbers[default: 50]')
cfgs = parser.parse_args()

save_path = 'tolerance'

V = 300
A = 12
D = 4
radius_list = [0.001 * x for x in range(51)]

def compute_point_dists(A, B):
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A - B, axis=-1)
    return dists

def manager(obj_name, pool_size=8):
    # load models
    label_path = '{}_labels.npz'.format(obj_name)
    label = np.load(os.path.join(cfgs.dataset_root, 'grasp_label', label_path))
    points = label['points']
    scores = label['scores']

    # create dict
    tolerance = mp.Manager().dict()
    dists = compute_point_dists(points, points)
    params = params = (scores, dists)

    # assign works
    pool = []
    process_cnt = 0
    work_list = [x for x in range(len(points))]
    for _ in range(pool_size):
        point_ind = work_list.pop(0)
        pool.append(mp.Process(target=worker, args=(obj_name, point_ind, params, tolerance)))
    [p.start() for p in pool]

    # refill
    while len(work_list) > 0:
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                point_ind = work_list.pop(0)
                p = mp.Process(target=worker, args=(obj_name, point_ind, params, tolerance))
                p.start()
                pool.append(p)
                process_cnt += 1
                print('{}/{}'.format(process_cnt, len(points)))
                break
    while len(pool) > 0:
        for ind, p in enumerate(pool):
            if not p.is_alive():
                pool.pop(ind)
                process_cnt += 1
                print('{}/{}'.format(process_cnt, len(points)))
                break

    # save tolerance
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    saved_tolerance = [None for _ in range(len(points))]
    for i in range(len(points)):
        saved_tolerance[i] = tolerance[i]
    saved_tolerance = np.array(saved_tolerance)
    np.save('{}/{}_tolerance.npy'.format(save_path, obj_name), saved_tolerance)

def worker(obj_name, point_ind, params, tolerance):
    scores, dists = params
    tmp_tolerance = np.zeros([V, A, D], dtype=np.float32)
    tic = time.time()
    for r in radius_list:
        dist_mask = (dists[point_ind] <= r)
        scores_in_ball = scores[dist_mask]
        pos_ratio = ((scores_in_ball > 0) & (scores_in_ball <= cfgs.mu_thresh)).mean(axis=0)
        tolerance_mask = (pos_ratio >= cfgs.pos_ratio_thresh)
        if tolerance_mask.sum() == 0:
            break
        tmp_tolerance[tolerance_mask] = r
    tolerance[point_ind] = tmp_tolerance
    toc = time.time()
    print("{}: point {} time".format(obj_name, point_ind), toc - tic)

if __name__ == '__main__':
    obj_list = ['%03d' % x for x in range(88)]
    for obj_name in obj_list:
        p = mp.Process(target=manager, args=(obj_name, cfgs.num_workers))
        p.start()
        p.join()