import numpy as np
from skimage.segmentation import slic
from skimage import io
from pathlib import Path

from utils import (
    extract_superpixel_features, compute_graph_matrices,
    clip_outliers, minmax_normalize, calculate_otsu_threshold, postprocess, performance
)
from Networks import train_model
from data_loader import load_data

# ==============================================================================
# DATASET SELECTION & HYPERPARAMETERS
# ==============================================================================

# 1. Global Defaults
base_lambdas = {
    'cross_lambda': 1.0, 'cycle_lambda': 1.0, 'structure_lambda': 1.0,
    'delt_lambda': 1.0, 'align_lambda': 1.0, 'smooth_lambda': 1.0
}
n_seg = 5000
cmp = 15
k_ratio = 0.1
base_hidden_dim = 16
base_knn_ratio = 0.3

# --- Dataset#1 ---
# structure_lambda: 5.0
# delt_lambda: [1.0, 2.0]
# cycle_lambda: ~1.0 - 2.0
data_name = 'Dataset#1'
base_lambdas['structure_lambda'] = 5.0
base_lambdas['delt_lambda'] = 2.0
base_lambdas['cycle_lambda'] = 2.0

# # --- Dataset#2 ---
# # structure_lambda: [1.0, 2.0]
# # delt_lambda: [1.0, 2.0]
# # cross/smooth: ~1.0
# data_name = 'Dataset#2'
# base_lambdas['structure_lambda'] = 2.0
# base_lambdas['delt_lambda'] = 2.0
# base_lambdas['cross_lambda'] = 1.0
# base_lambdas['smooth_lambda'] = 1.0

# # --- Dataset#3 ---
# # structure_lambda: [1.0, 2.0]
# # delt_lambda: [1.0, 2.0]
# # align_lambda: ~1.0
# # smooth_lambda: <= 1.0
# data_name = 'Dataset#3'
# base_lambdas['structure_lambda'] = 2.0
# base_lambdas['delt_lambda'] = 2.0
# base_lambdas['align_lambda'] = 1.0
# base_lambdas['smooth_lambda'] = 0.5

# # --- Dataset#4 ---
# # structure_lambda: [1.0, 2.0]
# # cross_lambda: ~1.0
# # smooth_lambda: <= 1.0
# data_name = 'Dataset#4'
# base_lambdas['structure_lambda'] = 2.0
# base_lambdas['cross_lambda'] = 1.0
# base_lambdas['smooth_lambda'] = 0.5

# # --- Dataset#5 ---
# # structure_lambda: [1.0, 2.0]
# # delt_lambda: [1.0, 2.0]
# # cycle/align: ~1.0
# data_name = 'Dataset#5'
# base_lambdas['structure_lambda'] = 2.0
# base_lambdas['delt_lambda'] = 1.0
# base_lambdas['cycle_lambda'] = 2.0
# base_lambdas['align_lambda'] = 2.0

# ==============================================================================
# MAIN PROCESS
# ==============================================================================
save_path = f'./results/{data_name}'
Path(save_path).mkdir(parents=True, exist_ok=True)

# Step 0: Load Data (Path logic is hidden in data_loader)
image_t1, image_t2, Ref_gt, imaget1t2 = load_data(data_name)

# Step 1: Feature Extraction Using Superpixels
sp_map = slic(imaget1t2, n_segments=n_seg, compactness=cmp)
sp_map = sp_map + 1
feats_pre, feats_post, scale_params = extract_superpixel_features(sp_map, image_t1, image_t2)

# Step 2: Construct Graphs
total_sp_count = feats_pre.shape[0]
feat_dim = feats_pre.shape[1]
Kmax = int(round(k_ratio * total_sp_count))

adj_mat_pre, lap_mat_pre, _ = compute_graph_matrices(feats_pre, Kmax)
adj_mat_post, lap_mat_post, _ = compute_graph_matrices(feats_post, Kmax)

training_config = {
    'dropout': 0.0, 'lr': 0.05, 'weight_decay': 0.0001, 'epoch': 300, 'num_heads': 4,
    'hidden_dim': base_hidden_dim,
    'knn_ratio': base_knn_ratio,
    **base_lambdas
}

# Step 3: Train
cycle_out_pre, rec_out_pre, delta_feat_pre, cycle_out_post, rec_out_post, delta_feat_post = train_model(
    feat_dim, total_sp_count, feats_pre, feats_post,
    adj_mat_pre, adj_mat_post, lap_mat_pre, lap_mat_post,
    n_seg, training_config
)

# Step 4: Compute Change Map
height, width = image_t1.shape[:2]
diff_map_pre = np.zeros((height, width))
diff_map_post = np.zeros((height, width))

for idx_node in range(total_sp_count):
    val_sq_pre = np.sum(np.square(delta_feat_pre[idx_node, :]))
    val_sq_post = np.sum(np.square(delta_feat_post[idx_node, :]))
    diff_map_pre[sp_map == idx_node + 1] = val_sq_pre
    diff_map_post[sp_map == idx_node + 1] = val_sq_post

dist_vec_pre = minmax_normalize(clip_outliers(diff_map_pre.reshape(-1, 1)))
dist_vec_post = minmax_normalize(clip_outliers(diff_map_post.reshape(-1, 1)))

merged_diff_vec = minmax_normalize(
    dist_vec_pre / np.mean(dist_vec_pre) + dist_vec_post / np.mean(dist_vec_post)
)

final_di_map = merged_diff_vec.reshape(height, width)
binary_thresh = calculate_otsu_threshold(merged_diff_vec)
change_binary_mask = np.zeros((height, width), dtype=np.uint8)
change_binary_mask[final_di_map > binary_thresh] = 255
change_binary_mask = postprocess(change_binary_mask)

# Step 5: Save
io.imsave(save_path + f'/DI.bmp', (final_di_map * 255).astype(np.uint8))
io.imsave(save_path + f'/ChangeMap.bmp', change_binary_mask.astype(np.uint8))
visual_overlay = np.stack((Ref_gt, np.squeeze(change_binary_mask), Ref_gt), 2)
io.imsave(save_path + f'/fccImage.bmp', visual_overlay.astype(np.uint8))

# Step 6: Evaluate
fp, fn, OE, pcc, kappa = performance(change_binary_mask, Ref_gt)
with open(Path(save_path, "evaluate.txt"), "a") as f:
    f.write(f'FP: {fp}\nFN: {fn}\nOE: {OE}\nOA: {pcc}\nKC (kappa): {kappa}\n')
