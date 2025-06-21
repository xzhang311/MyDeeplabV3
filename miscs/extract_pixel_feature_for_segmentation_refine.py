import argparse
import os
import cv2
import pickle
import numpy as np
from multiprocessing import Pool

def get_file_names_in_dir(dir):
    onlyfiles = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    return onlyfiles

def produce_pos(Rs, Cs, rows, cols, win_size = 3, nsamples = 10):
    Rs = np.tile(Rs, nsamples)
    Cs = np.tile(Cs, nsamples)

    l = len(Rs)

    # offsize = np.random.uniform(low=-1 * win_size, high=win_size, size=l).astype(np.int)
    offsize = np.random.normal(loc = 0, scale = win_size, size = l).astype(np.int)
    Rs = np.clip(Rs + offsize, 0, rows - 1)
    # offsize = np.random.uniform(low=-1 * win_size, high=win_size, size=l).astype(np.int)
    offsize = np.random.normal(loc = 0, scale = win_size, size = l).astype(np.int)
    Cs = np.clip(Cs + offsize, 0, cols - 1)

    rst = np.zeros([l, 2])
    rst[:, 0] = Rs
    rst[:, 1] = Cs
    return rst

def sampleing_around_nonzero_gradient(gt_mask, win_size = 3, nsamples = 10):
    sobelx = cv2.Sobel(gt_mask, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gt_mask, cv2.CV_64F, 0, 1, ksize=5)
    grad = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    Rs, Cs = np.where(grad > 0)
    rst = produce_pos(Rs, Cs, gt_mask.shape[0], gt_mask.shape[1], win_size, nsamples)
    return rst

def sampling_around_discrepency_areas(gt_mask, pred_mask, win_size = 3, nsamples = 10):
    img_diff = np.abs(gt_mask.astype(np.float) - pred_mask.astype(np.float))
    Rs, Cs = np.where(img_diff > 0)
    rst = produce_pos(Rs, Cs, gt_mask.shape[0], gt_mask.shape[1], win_size, nsamples)
    return rst

def sampling_around_low_confidence_area(pred_mask, win_size = 3, nsamples = 10, threshold = 0.48):
    img_tmp0 = pred_mask.astype(float) > 0.2
    img_tmp1 = pred_mask.astype(float) < 1
    img = cv2.bitwise_and(img_tmp0.astype(np.uint8), img_tmp1.astype(np.uint8))
    Rs, Cs = np.where(img == 1)
    rst = produce_pos(Rs, Cs, pred_mask.shape[0], pred_mask.shape[1], win_size, nsamples)
    return rst

def dedup(pos, rows, cols):
    z = np.zeros([rows, cols])
    pos = np.asarray(pos).astype(int)
    z[pos[:, 0], pos[:, 1]] = 1
    Rs, Cs = np.where(z==1)
    rst = np.zeros([len(Rs), 2])
    rst[:, 0] = Rs
    rst[:, 1] = Cs
    return list(rst)

def visualize_sampling_pos(mask_dir, output_dir, rows, cols, mask_name, pos):
    mask = cv2.imread(os.path.join(mask_dir, mask_name))
    mask = cv2.resize(mask, (cols, rows))
    pos = np.asarray(pos).astype(int)
    mask[pos[:, 0], pos[:, 1], :] = [0, 0, 255]
    cv2.imwrite(os.path.join(output_dir, mask_name), mask)

def process_single_mask(dict):
    mask_dir = dict['mask_dir']
    pre_rst_dir = dict['pre_rst_dir']
    output_dir = dict['output_dir']
    mask_name = dict['mask_name']

    sampling_pos = []
    id, ext = os.path.splitext(mask_name)
    id = id[:-5]

    mask_name_pred = id + '_mask.png'
    feat_name_pred = id + '_features.pkl'

    # load ground truth mask
    gt_mask = cv2.imread(os.path.join(mask_dir, mask_name), 0) / 255.0
    # load predicted mask
    pred_mask = cv2.imread(os.path.join(pre_rst_dir, mask_name_pred), 0) / 255.0
    # load feature map
    feat = pickle.load(open(os.path.join(pre_rst_dir, feat_name_pred), "rb"))

    pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))

    # sample 5 times at each poi
    nsamples = 10
    win_size = 5

    rst = sampleing_around_nonzero_gradient(gt_mask, win_size=win_size, nsamples=nsamples)
    sampling_pos.extend(rst)
    rst = sampleing_around_nonzero_gradient(pred_mask, win_size=win_size, nsamples=nsamples)
    sampling_pos.extend(rst)
    rst = sampling_around_discrepency_areas(gt_mask, pred_mask, win_size=win_size, nsamples=nsamples)
    sampling_pos.extend(rst)
    rst = sampling_around_low_confidence_area(pred_mask, win_size=win_size, nsamples=nsamples)
    sampling_pos.extend(rst)

    sampling_pos = dedup(sampling_pos, pred_mask.shape[0], pred_mask.shape[1])

    visualize_sampling_pos(pre_rst_dir, output_dir, gt_mask.shape[0], gt_mask.shape[1], mask_name, sampling_pos)

    f = open(os.path.join(output_dir, id + '.pkl'), 'wb')
    pickle.dump(sampling_pos, f)

    print('Mask name :{} Len: {}'.format(mask_name, len(sampling_pos)))

def process(mask_dir, pre_rst_dir, output_dir):
    mask_names = get_file_names_in_dir(mask_dir)

    existing_mask_names = get_file_names_in_dir(output_dir)

    dicts = []

    for mask_name in mask_names:
        if mask_name in existing_mask_names:
            continue

        dict = {}
        dict['mask_dir'] = mask_dir
        dict['pre_rst_dir'] = pre_rst_dir
        dict['output_dir'] = output_dir
        dict['mask_name'] = mask_name
        dicts.append(dict)


    pool = Pool(16)
    pool.map(process_single_mask, dicts)

    pool.close()
    pool.join()
    return 1

def main():
    parser = argparse.ArgumentParser(description='Extract pixel features for segmentation refine')
    # parser.add_argument('--mask_dir', help='Dir of ground truth mask')
    # parser.add_argument('--pred_rst_dir', help='Dir of predicted masks and features')
    # parser.add_argument('--output_dir', help='Dir of output')
    args = parser.parse_args()

    args.mask_dir = '/mnt/ebs_xizhn2/Data/Segmentation/golden_set/masks'
    args.pred_rst_dir = '/mnt/ebs_xizhn2/Data/Segmentation/golden_set/masks_generated_by_old_model'
    args.output_dir = '/mnt/ebs_xizhn2/Data/Segmentation/golden_set/pixel_features_by_old_model'

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    process(args.mask_dir, args.pred_rst_dir, args.output_dir)

if __name__ == '__main__':
    main()