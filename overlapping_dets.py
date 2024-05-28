#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:04:55 2024

@author: rbain
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import square
from skimage.segmentation import flood

# this script will process every PNG image in this directory
working_dir = r'/home/rbain/Desktop/ELA_detonator_overlap/split_crops'

all_files = os.listdir(working_dir)
tifs = [f for f in all_files if f.endswith(".tif")]
tifs = sorted(tifs)

assert len(tifs) > 0, f'{working_dir} contains no TIF images :('

# 300 for 'CROP_right_CA3_6L_tiled_first_z_snap.tif'
r_threshold = 300 
g_threshold = 300

for ch1, ch2 in zip(tifs[::2], tifs[1::2]):
    err_txt = "filename structure violated"
    assert ch1[-5] == '1' and ch2[-5] == '2', err_txt
    assert ch1[::-5] == ch2[::-5], err_txt

    fig, ax = plt.subplots(ncols=2)
    
    ch1_fullpath = os.path.join(working_dir, ch1)
    og_red = cv2.imread(ch1_fullpath, cv2.IMREAD_UNCHANGED)
    print(f'\ndtype: {og_red.dtype}, shape: {og_red.shape}')
    print(f'min: {np.min(og_red)}, max: {np.max(og_red)}')
    red = og_red.copy()
    
    ch2_fullpath = os.path.join(working_dir, ch2)
    og_green = cv2.imread(ch2_fullpath, cv2.IMREAD_UNCHANGED)
    print(f'\ndtype: {og_green.dtype}, shape: {og_green.shape}')
    print(f'min: {np.min(og_green)}, max: {np.max(og_green)}')
    green = og_green.copy()
    
    red[og_red < r_threshold] = 0
    red[og_green < g_threshold] = 0
    green[og_green < g_threshold] = 0
    green[og_red < r_threshold] = 0
    
    og_red = og_red / og_red.max()
    og_green = og_green / og_green.max()
    
    red = red / red.max() # normalize
    green = green / green.max() # normalize
    ax[0].imshow(red)
    ax[1].imshow(green)
    ax[0].set_title('red')
    ax[1].set_title('green')
    
    fig, ax = plt.subplots(ncols=2)
    
    gain = 3
    blue = np.zeros_like(og_green)
    thresholded = np.dstack([red, green, blue])
    ax[0].imshow(thresholded * gain)
    ax[0].set_title('thresholded')
    
    original = np.dstack([og_red, og_green, blue])
    ax[1].imshow(original * gain)
    ax[1].set_title('original')
    
    clusters = []
    binary_thresh = thresholded.copy()[..., 0]
    binary_thresh[binary_thresh > 0] = 1
    H, W = binary_thresh.shape
    # go through every pixel
    for h in range(H):
        for w in range(W):
            # flood fill if not zero
            if binary_thresh[h, w] != 0:
                assert binary_thresh[h, w] == 1, "non-binary values detected"
                cluster_mask = flood(binary_thresh, (h, w), footprint=square(3))
                # record the current cluster
                clusters.append(cluster_mask)
                # set cluster's pixels to black
                binary_thresh[cluster_mask == True] = 0
    
    fig, ax = plt.subplots(ncols=2)
    og_binary_thresh = thresholded.copy()[..., 0]
    og_binary_thresh[og_binary_thresh > 0] = 1
    
    ax[0].imshow(og_binary_thresh, cmap='gray')
    ax[0].set_title('binary thresholding')
    
    print(f'\n{len(clusters)} clusters before filtering by size')
    # filter by size
    size_thresh = 150
    clusters = [c for c in clusters if c.sum() > size_thresh]
    print(f'{len(clusters)} clusters after filtering by size')
    
    ### @TODO 
    # print statistics about clusters
    # save statistics about clusters
    
    colors = [.25, .5, .75, 1]
    for i, c in enumerate(clusters):
        binary_thresh[c == True] = colors[i % len(colors)]
    
    ax[1].imshow(binary_thresh, cmap='jet', interpolation='none')
    ax[1].set_title('clusters')
    
    fig, ax = plt.subplots(ncols=2)
    
    original = np.dstack([og_red, og_green, blue])
    ax[1].imshow(original * gain)
    ax[1].set_title('original')
    
    clusters_removed = original.copy()
    for c in clusters:
        clusters_removed[c == True] = 0
    ax[0].imshow(clusters_removed * gain)
    ax[0].set_title('clusters removed')
    
    plt.show()
















'''
for png in PNGs:
    original = plt.imread(png)
    thresholded = original.copy()

    fig, ax = plt.subplots(ncols=1)
    ax.imshow(original)
    ax.set_title('Original')
    plt.show()
    
    fig, ax = plt.subplots(ncols=1)
    # RGB channel format
    red = original[..., 0].flatten()
    green = original[..., 1].flatten()
    ax.hist(red, color='r', bins=256)
    ax.hist(green, color='g', bins=256)
    ax.set_title('red and green histograms')
    r_min = red.min();   r_max = red.max()
    g_min = green.min(); g_max = green.max()
    print(r_min, r_max, g_min, g_max)
    plt.show()
    
    r_threshold = _r_check = input("\nWhat threshold for red would you like to use? ")
    if r_threshold.count('.') < 2:
        _r_check = r_threshold.replace('.', '1')
    assert _r_check.isnumeric(), "Your input was not a number :("
    r_threshold = float(r_threshold)
    assert r_threshold <= r_max, f'{r_threshold} is larger than the largest value of {r_max}'
    assert r_threshold >= r_min, f'{r_threshold} is smaller than the smallest value of {r_min}'
    
    g_threshold = _g_check = input("\nWhat threshold for green would you like to use? ")
    if g_threshold.count('.') < 2:
        _g_check = g_threshold.replace('.', '1')
    assert _g_check.isnumeric(), "Your input was not a number :("
    g_threshold = float(g_threshold)
    assert g_threshold <= g_max, f'{g_threshold} is larger than the largest value of {g_max}'
    assert g_threshold >= g_min, f'{g_threshold} is smaller than the smallest value of {g_min}'
    
    red[red < r_threshold] = 0 
    red[green < g_threshold] = 0 
    thresholded[..., 0] = red.reshape(thresholded[..., 0].shape)
    green[green < g_threshold] = 0 
    green[red < r_threshold] = 0 
    thresholded[..., 1] = green.reshape(thresholded[..., 1].shape)
    
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(original)
    ax[0].set_title('Original')
    ax[1].imshow(thresholded)
    # ax[1].plot(76, 76, 'wo')  # seed point
    ax[1].set_title('Thresholded')
    
    plt.imsave(r'/tmp/test2.png', thresholded)
    
    plt.show()
'''