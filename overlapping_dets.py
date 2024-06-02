import os
import csv
import cv2
import time
import datetime
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from skimage.morphology import square
from skimage.segmentation import flood


r_threshold = 300 
g_threshold = 300
def img_worker(ch1, ch2, results_dir, results_dict):
    '''
    Image processing function to be ran in parallel.
    Takes separate red (PSD95) and green (synaptotagmin 1) channels.
    Saves images in results_dir. Returns info thru results_dict
    '''
    
    # naming constraint: Begins w/ # of pyr cells followed by '_'
    char_after_soma_count = ch1.find("_") + 1
    name_root = ch1[char_after_soma_count:-7]
    results_dict['name_root'] = name_root
    
    ch1_fullpath = os.path.join(tif_dir, ch1)
    og_red = cv2.imread(ch1_fullpath, cv2.IMREAD_UNCHANGED)
    red = og_red.copy()
    ch2_fullpath = os.path.join(tif_dir, ch2)
    og_green = cv2.imread(ch2_fullpath, cv2.IMREAD_UNCHANGED)
    green = og_green.copy()
    
    # for ch in [og_red, og_green]:
    #     print(f'\ndtype: {ch.dtype}, shape: {ch.shape}')
    #     print(f'min: {np.min(ch)}, max: {np.max(ch)}')
    
    red[og_red < r_threshold] = 0
    red[og_green < g_threshold] = 0
    green[og_green < g_threshold] = 0
    green[og_red < r_threshold] = 0
    
    red = red / red.max() # normalize
    green = green / green.max() # normalize
    
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    
    thresholded = np.dstack([red, green, np.zeros_like(og_green)])
    
    clusters = []
    size_thresh = 150
    binary_thresh = thresholded.copy()[..., 0]
    binary_thresh[binary_thresh > 0] = 1
    H, W = binary_thresh.shape
    def revert_mask(cluster_m):
        # masks are the same shape of the image normally
        # saving just positions and then reverting saves a lot of memory
        mask = np.zeros((H,W), dtype=bool)
        for i,j in cluster_m:
            mask[i][j] = True
        return mask            
    # Now go through every 16th pixel...
    last_p = 0
    sq = square(3)
    # cheeky little trick to 4X the speed
    for h in range(0, H, 3):
        percent = int(h / H * 100)
        if last_p != percent:    
            print(f"{name_root} is {percent}% flooded")
            last_p = percent
        # that's 16x savings for those keeping track at home!
        for w in range(0, W, 4):
            # flood fill if not zero
            if binary_thresh[h, w] != 0:
                cluster_mask = flood(binary_thresh, (h, w), footprint=sq)
                # set cluster's pixels to black
                binary_thresh[cluster_mask == True] = 0
                # filter by size
                if cluster_mask.sum() >= size_thresh:
                    cm = np.argwhere(cluster_mask)
                    # record the current cluster
                    clusters.append(cm)
                del cluster_mask
    
    og_binary_thresh = thresholded.copy()[..., 0]
    og_binary_thresh[og_binary_thresh > 0] = 1
    del thresholded
    
    ax[0,0].imshow(og_binary_thresh, cmap='gray')
    ax[0,0].set_title('binary thresholding')
    del og_binary_thresh
    
    colors = [.25, .5, .75, 1]
    for i, c in enumerate(clusters):
        c = revert_mask(c)
        binary_thresh[c == True] = colors[i % len(colors)]
    
    ax[0,1].imshow(binary_thresh, cmap='jet', interpolation='none')
    ax[0,1].set_title('clusters')
    
    gain = 3
    norm_red = og_red / og_red.max()
    norm_green = og_green / og_green.max()
    original = np.dstack([norm_red, norm_green, np.zeros_like(og_green)])
    ax[1,1].imshow(original * gain)
    ax[1,1].set_title('original')
    del norm_red, norm_green
    
    clusters_removed = original.copy()
    for c in clusters:
        c = revert_mask(c)
        clusters_removed[c == True] = 0
    ax[1,0].imshow(clusters_removed * gain)
    ax[1,0].set_title('clusters removed')
    
    results_dict['pyr_cell_count'] = int(ch1[:ch1.find("_")])
    
    cluster_img_path = f'{name_root}.png'
    cluster_img_path = os.path.join(results_dir, cluster_img_path)
    plt.savefig(cluster_img_path, dpi=300)
    plt.close(fig) # prevent plotting huge figures inline
    
    # calculate actual non-blank space in image
    combined = np.multiply(og_red, og_green)
    combined[combined != 0] = 1
    true_img_area = combined.sum()
    results_dict['image_area'] = true_img_area
    
    results = []
    total_psd = 0
    total_syn = 0
    total_size = 0
    for i, c in enumerate(clusters):
        c = revert_mask(c)
        row = [i]
        cluster_size = c.sum()
        row.append(cluster_size)
        total_size += cluster_size
        cluster_psd = og_red[c == True]
        avg_psd = cluster_psd.mean()
        row.append(avg_psd)
        cluster_syn = og_green[c == True]
        avg_syn = cluster_syn.mean()
        row.append(avg_syn)
        results.append(row)
        total_psd += cluster_psd.sum()
        total_syn += cluster_syn.sum()
        
    n_clusters = len(results)
    # Avg pixel values over all clusters in a given image
    avg_psd = total_psd / total_size
    avg_syn = total_syn / total_size
    avg_size = total_size / n_clusters
    results_dict["avg_psd"] = avg_psd
    results_dict["avg_syn"] = avg_syn
    results_dict["avg_size"] = avg_size
    results_dict["n_clusters"] = n_clusters
    
    # save cluster results in individual csv files too
    csv_name = os.path.join(results_dir, f"{name_root}.csv")
    # writing to csv file
    with open(csv_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        header = ['Cluster ID', 'Size', 'Avg PSD', 'Avg Syn1']
        writer.writerow(header)
        writer.writerows(results)

if __name__ == "__main__":
    
    tif_dir = os.getcwd()
    all_files = os.listdir(tif_dir)
    tifs = [f for f in all_files if f.endswith(".tif")]
    tifs = sorted(tifs)
    err_txt = 'Your current directory contains no TIF images :('
    assert len(tifs) > 0, err_txt
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(tif_dir, 'results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    processing_start = time.time()
    
    image_workers = []
    worker_results = []
    manager = multiprocessing.Manager()
    for ch1, ch2 in zip(tifs[::2], tifs[1::2]):
        err_txt = "filename structure violated"
        assert ch1[-5] == '1' and ch2[-5] == '2', err_txt
        assert ch1[:-5] == ch2[:-5], err_txt

        shared_dict = manager.dict()
        worker_results.append(shared_dict)
        args = {
            'ch1': ch1, 
            'ch2': ch2, 
            'results_dir': results_dir,
            'results_dict': worker_results[-1]
        }
        # process each image simultaneously in parallel processes
        w = multiprocessing.Process(target=img_worker, kwargs=args)
        w.start()
        image_workers.append(w)
        # prevent OOM, limits # of parallel processes
        if len(image_workers) % 6 == 0:
            for w in image_workers:
                w.join()
                w.close()
            image_workers = []
    
    # wait for the processes to finish
    for w in image_workers:
        w.join()
        w.close()
        
    # save master csv file
    combined_results = []
    combined_csv = os.path.join(results_dir, "combined_results.csv")
    with open(combined_csv, 'w') as main_csv_file:
        main_writer = csv.writer(main_csv_file)
        header = [
            'Name', '# Clusters', 'Avg Size', 'Avg PSD',
            'Avg Syn1', '# Cells', 'Effective Image Area in Pixels'
        ]
        main_writer.writerow(header)
        for results in worker_results:
            name = results['name_root']
            n_cells = results['pyr_cell_count']
            img_area = results["image_area"]
            n_clusters = results['n_clusters']
            avg_psd = results['avg_psd']
            avg_syn = results['avg_syn']
            avg_size = results['avg_size']
            # singular result for this whole slice, for the combined csv
            result = [name, n_clusters, avg_size, avg_psd, avg_syn, n_cells, img_area]
            combined_results.append(result)
        main_writer.writerows(combined_results)
    
    print(f'\nCompleted in {(time.time() - processing_start) / 60:.2f} minutes.')
    print(f'\nResults saved in {results_dir}')
