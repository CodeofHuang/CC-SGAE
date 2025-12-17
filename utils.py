import numpy as np
from skimage import measure
import cv2
import torch
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from collections import Counter


def convert_image(image):
    if image is not None:
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"Unable to read image")
    return image


def get_superpixel_indices(segment_map):
    max_label = np.max(segment_map)
    indices_list = [np.where(segment_map.ravel() == label)[0] for label in range(1, max_label + 1)]
    return indices_list


def extract_superpixel_features(segment_map, img_pre, img_post):
    height, width, bands_pre = img_pre.shape
    _, _, bands_post = img_post.shape

    sp_indices = get_superpixel_indices(segment_map)
    num_segments = segment_map.max()

    flat_pre = img_pre.reshape((height * width, bands_pre))
    flat_post = img_post.reshape((height * width, bands_post))

    feats_pre = np.zeros((num_segments, 2 * bands_pre))
    feats_post = np.zeros((num_segments, 2 * bands_post))

    for idx in range(num_segments):
        curr_indices = sp_indices[idx]

        pixels_pre = flat_pre[curr_indices, :]
        avg_pre = np.mean(pixels_pre, axis=0)
        med_pre = np.median(pixels_pre, axis=0)
        # var_pre = np.var(pixels_pre, axis=0)
        feats_pre[idx, :] = np.concatenate([avg_pre, med_pre], axis=0)

        pixels_post = flat_post[curr_indices, :]
        avg_post = np.mean(pixels_post, axis=0)
        med_post = np.median(pixels_post, axis=0)
        # var_post = np.var(pixels_post, axis=0)
        feats_post[idx, :] = np.concatenate([avg_post, med_post], axis=0)

    feats_pre = feats_pre.T
    feats_post = feats_post.T

    feats_pre = np.nan_to_num(feats_pre)
    feats_post = np.nan_to_num(feats_post)

    scaling_factors = np.zeros(bands_pre + bands_post)
    for b in range(bands_pre):
        scaling_factors[b] = np.max(feats_pre[b, :]) + np.finfo(float).eps
    for b in range(bands_post):
        scaling_factors[bands_pre + b] = np.max(feats_post[b, :]) + np.finfo(float).eps

    max_val_pre = np.max(feats_pre, axis=1, keepdims=True) + np.finfo(float).eps
    feats_pre = feats_pre / max_val_pre

    max_val_post = np.max(feats_post, axis=1, keepdims=True) + np.finfo(float).eps
    feats_post = feats_post / max_val_post

    feats_pre = feats_pre.T
    feats_post = feats_post.T

    return feats_pre, feats_post, scaling_factors


def compute_graph_matrices(features, neighbors_k=None):
    if neighbors_k is None:
        neighbors_k = int(np.round(np.sqrt(features.shape[0])))

    if features.ndim == 1:
        features = np.expand_dims(features, axis=0)

    num_samples = features.shape[0]
    sim_matrix = np.zeros((num_samples, num_samples))
    search_k = neighbors_k + 1

    knn_model = NearestNeighbors(n_neighbors=search_k).fit(features)
    dists, neighbor_indices = knn_model.kneighbors(features)

    deg_counts = Counter(neighbor_indices.ravel())
    k_vec = np.array([count for _, count in deg_counts.items()])

    limit_max = neighbors_k
    limit_min = int(np.round(limit_max / 10)) + 1
    k_vec[k_vec >= limit_max] = limit_max
    k_vec[k_vec <= limit_min] = limit_min

    gaussian_kernel = lambda d: np.exp(-d ** 2 / 2)
    for idx in range(num_samples):
        local_dists = dists[idx, 1:]
        weights = gaussian_kernel(local_dists)
        sim_matrix[idx, neighbor_indices[idx, 1:]] = weights

    adj_mat = (sim_matrix + sim_matrix.T) / 2
    adj_mat[adj_mat != 0] = 1
    adj_sparse = sp.coo_matrix(adj_mat)

    lap_sparse = sp.csgraph.laplacian(adj_sparse, normed=True)

    adj_row = torch.from_numpy(adj_sparse.row).long()
    adj_col = torch.from_numpy(adj_sparse.col).long()
    adj_tensor = torch.stack([adj_row, adj_col], dim=0)

    lap_tensor = torch.from_numpy(lap_sparse.todense()).float()

    return adj_tensor, lap_tensor, sim_matrix


def clip_outliers(data_array):
    arr_copy = np.copy(data_array)
    mu = np.mean(data_array)
    sigma = np.std(data_array)

    limit = mu + 4 * sigma
    mask = data_array > limit

    arr_copy[mask] = np.max(arr_copy[~mask])
    return arr_copy


def minmax_normalize(input_arr):
    min_val = input_arr.min()
    max_val = input_arr.max()
    denominator = max_val - min_val + np.finfo(float).eps
    normalized = (input_arr - min_val) / denominator
    return normalized


def calculate_otsu_threshold(values, steps=1000):
    val_max = np.max(values)
    val_min = np.min(values)
    count = values.shape[0]

    step_size = (val_max - val_min) / steps
    curr_thresh = val_min + step_size

    optimal_thresh = val_min
    max_variance = 0

    while curr_thresh <= val_max:
        group1 = values[values < curr_thresh]
        group2 = values[values >= curr_thresh]

        weight1 = group1.shape[0] / count
        weight2 = group2.shape[0] / count

        avg1 = group1.mean() if group1.size > 0 else 0
        avg2 = group2.mean() if group2.size > 0 else 0

        curr_variance = weight1 * weight2 * np.power((avg1 - avg2), 2)

        if max_variance < curr_variance:
            max_variance = curr_variance
            optimal_thresh = curr_thresh

        curr_thresh += step_size

    return optimal_thresh


def postprocess(res):
    res_new = res
    res = measure.label(res, connectivity=2)
    num = res.max()
    for i in range(1, num + 1):
        idy, idx = np.where(res == i)
        if len(idy) <= 20:
            res_new[idy, idx] = 0
    return res_new


def performance(cm, gt):
    TP, TN, FP, FN = 0, 0, 0, 0
    pred_label_data = np.reshape(cm, (1, -1))
    true_label_data = np.reshape(gt, (1, -1))
    all_num = pred_label_data.shape[1]

    for i in range(all_num):
        if pred_label_data[0][i] == 255 and true_label_data[0][i] == 255:
            TP += 1
        elif pred_label_data[0][i] == 255 and true_label_data[0][i] == 0:
            FP += 1
        elif pred_label_data[0][i] == 0 and true_label_data[0][i] == 255:
            FN += 1
        else:
            TN += 1

    OE = FN + FP
    PCC = (TP + TN) / (TP + FP + TN + FN)
    PRE = ((TP + FP) * (TP + FN)) / ((TP + TN + FP + FN) ** 2) + ((FN + TN) * (FP + TN)) / ((TP + TN + FP + FN) ** 2)
    KC = (PCC - PRE) / (1 - PRE)

    return FP, FN, OE, PCC, KC


def image_normlized(img, norm_type):
    if norm_type == 'sar':

        positive_values = img[np.abs(img) > 0]
        if positive_values.size > 0:
            min_nonzero = np.min(positive_values)
            img[np.abs(img) <= 0] = min_nonzero

        img = np.log(img + 1)

    img_height, img_width, channel = img.shape
    img = img.reshape(-1, channel)

    max_value = np.max(img, axis=0, keepdims=True)
    min_value = np.min(img, axis=0, keepdims=True)
    diff_value = max_value - min_value + np.finfo(float).eps
    nm_img = (img - min_value) / diff_value
    nm_img = nm_img.reshape(img_height, img_width, channel)

    return nm_img
