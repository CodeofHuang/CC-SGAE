import cv2
import numpy as np
import imageio.v2 as imageio
from utils import convert_image, image_normlized


def get_dataset_config(data_name):
    root_dir = f'../datasets/{data_name}'
    config = {
        'im1_path': '', 'im2_path': '', 'imgt_path': '',
        'im1_type': 'opt', 'im2_type': 'opt',
        'use_cv2_loader': False,
        'apply_mean_filter': True
    }

    # --- Dataset Specific Configurations ---
    if data_name == 'Dataset#1':
        config['im1_path'] = f'{root_dir}/im1.jpg'
        config['im2_path'] = f'{root_dir}/im2.jpg'
        config['imgt_path'] = f'{root_dir}/im3.bmp'
        config['im1_type'] = 'sar'
        config['im2_type'] = 'opt'

    elif data_name == 'Dataset#2':
        config['im1_path'] = f'{root_dir}/im1.png'
        config['im2_path'] = f'{root_dir}/im2.png'
        config['imgt_path'] = f'{root_dir}/im3.png'
        config['im1_type'] = 'opt'
        config['im2_type'] = 'sar'

    elif data_name == 'Dataset#3':
        config['im1_path'] = f'{root_dir}/im1.tif'
        config['im2_path'] = f'{root_dir}/im2.tif'
        config['imgt_path'] = f'{root_dir}/im3.tif'
        config['im1_type'] = 'opt'
        config['im2_type'] = 'sar'
        config['use_cv2_loader'] = True

    elif data_name == 'Dataset#4':
        config['im1_path'] = f'{root_dir}/im1.png'
        config['im2_path'] = f'{root_dir}/im2.png'
        config['imgt_path'] = f'{root_dir}/im3.png'
        config['im1_type'] = 'sar'
        config['im2_type'] = 'sar'

    elif data_name == 'Dataset#5':
        config['im1_path'] = f'{root_dir}/im1.bmp'
        config['im2_path'] = f'{root_dir}/im2.bmp'
        config['imgt_path'] = f'{root_dir}/im3.bmp'
        config['im1_type'] = 'opt'
        config['im2_type'] = 'opt'
        config['apply_mean_filter'] = False

    else:
        # Fallback for generic bmp datasets
        config['im1_path'] = f'{root_dir}/im1.bmp'
        config['im2_path'] = f'{root_dir}/im2.bmp'
        config['imgt_path'] = f'{root_dir}/im3.bmp'

    return config


def load_data(data_name):

    cfg = get_dataset_config(data_name)

    if cfg['use_cv2_loader']:
        image_t1 = cv2.imread(cfg['im1_path'], cv2.IMREAD_UNCHANGED)
        image_t2 = cv2.imread(cfg['im2_path'], cv2.IMREAD_UNCHANGED)
        Ref_gt = cv2.imread(cfg['imgt_path'], cv2.IMREAD_UNCHANGED)
        image_t1 = convert_image(image_t1)
        image_t2 = convert_image(image_t2)
        Ref_gt = convert_image(Ref_gt)
    else:
        image_t1 = imageio.imread(cfg['im1_path'])
        image_t2 = imageio.imread(cfg['im2_path'])
        Ref_gt = imageio.imread(cfg['imgt_path'])

    image_t1 = image_normlized(image_t1.astype(np.float32), cfg['im1_type'])
    image_t2 = image_normlized(image_t2.astype(np.float32), cfg['im2_type'])

    if cfg['apply_mean_filter']:
        h = np.ones((5, 5), np.float32) / 25
        image_t1 = cv2.filter2D(image_t1, -1, h, borderType=cv2.BORDER_REFLECT)
        image_t2 = cv2.filter2D(image_t2, -1, h, borderType=cv2.BORDER_REFLECT)

    height, width, _ = image_t1.shape
    image_t1_single = np.mean(image_t1, axis=-1, keepdims=True)
    image_t2_single = np.mean(image_t2, axis=-1, keepdims=True)
    zero_channel = np.zeros((height, width, 1))
    imaget1t2 = np.concatenate([image_t1_single, image_t2_single, zero_channel], axis=-1)

    return image_t1, image_t2, Ref_gt, imaget1t2