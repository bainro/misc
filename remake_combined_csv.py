# Made this script because I'm having to run this across multiple computers,
# and also sometimes it crashes. This script will re-make combined.csv
import os
import csv
import cv2
import numpy as np

if __name__ == "__main__":
    print()
    print("meant to be ran from the place combined.csv would usually be found")
    print("i.e. in the results subdirectory, 2 directories down from this script")
    print()
    results_dir = os.getcwd()
    tif_dir = os.path.join(results_dir, os.pardir, os.pardir)
    tif_dir = os.path.abspath(tif_dir)
    all_files = os.listdir(tif_dir)
    tifs = [f for f in all_files if f.endswith(".tif")]
    tifs = sorted(tifs)
    err_txt = f'{tif_dir} contains no TIF images :('
    assert len(tifs) > 0, err_txt
    
    rows = []
    for ch1, ch2 in zip(tifs[::2], tifs[1::2]):
        err_txt = "filename structure violated"
        assert ch1[-5] == '1' and ch2[-5] == '2', err_txt
        assert ch1[:-5] == ch2[:-5], err_txt

        # naming constraint: Begins w/ # of pyr cells followed by '_'
        char_after_soma_count = ch1.find("_") + 1
        name_root = ch1[char_after_soma_count:-7]
        img_csv = os.path.join(f'{name_root}.csv')
        if not os.path.isfile(img_csv):
            print(f"{img_csv} not found! Skipping.")
            continue
        rows.append([name_root])    
        
        ch1_fullpath = os.path.join(tif_dir, ch1)
        ch2_fullpath = os.path.join(tif_dir, ch2)
        og_red = cv2.imread(ch1_fullpath, cv2.IMREAD_UNCHANGED)
        og_green = cv2.imread(ch2_fullpath, cv2.IMREAD_UNCHANGED)
        
        # load individual csv files
        clusters = []
        with open(img_csv, mode ='r') as file:
            csv_file = csv.reader(file)
            for i, line in enumerate(csv_file):
                # omit header & empty lines
                if i == 0 or line == []:
                    continue
                clusters.append(line)
        
        n_clusters = len(clusters)
        rows[-1].append(n_clusters)
        
        total_psd = 0
        total_syn = 0
        total_size = 0
        # c = [cluster id, size, psd, syn]
        for i, c in enumerate(clusters):
            cluster_size = int(c[1])
            total_size += cluster_size
            total_psd += float(c[2]) * cluster_size
            total_syn += float(c[3]) * cluster_size
        
        # Avg pixel values over all clusters in a given image
        avg_psd = total_psd / total_size
        avg_syn = total_syn / total_size
        avg_size = total_size / n_clusters
        rows[-1].append(avg_size)
        rows[-1].append(avg_psd)
        rows[-1].append(avg_syn)
        
        pyr_cell_count = int(ch1[:ch1.find("_")])
        rows[-1].append(pyr_cell_count)
        
        # calculate actual non-blank space in image
        combined = np.multiply(og_red, og_green)
        combined[combined != 0] = 1
        img_area = combined.sum()
        rows[-1].append(img_area)
        
    # save master csv file
    combined_csv = os.path.join(results_dir, "combined_results.csv")
    with open(combined_csv, 'w') as main_csv_file:
        main_writer = csv.writer(main_csv_file)
        header = [
            'Name', '# Clusters', 'Avg Size', 'Avg PSD',
            'Avg Syn1', '# Cells', 'Effective Image Area in Pixels'
        ]
        main_writer.writerow(header)
        main_writer.writerows(rows)