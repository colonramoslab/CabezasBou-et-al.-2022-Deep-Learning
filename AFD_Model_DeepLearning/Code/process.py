

import tifffile as tiff
import csv
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, img_as_ubyte
from scipy import misc, spatial
from skimage.transform import rescale, resize
import os
import time
import sys
import pandas as pd

np.seterr(all='ignore')

def get_properties(CCs):

    props = measure.regionprops(CCs)
    volumes = np.array([prop.area for prop in props])
    bounding_boxes = np.array([prop.bbox for prop in props])

    return volumes, bounding_boxes


if __name__ == '__main__':
    
    home = 'E:\\FreelyMoving\\DeepLearning\\AFD_Model\\Data\\9_15_19\\Tc20\\'

    # Remove debris hyperparameters:
    vol_thresh_rd = 100

    # Bound fixer. 
    x_edge = 50
    y_edge = 25

    # Erosions. 
    num_erosion_dilation = 12

    # No. dilations to fill out neuron
    neuron_dilates = 5

    # Ratio
    ratio_thresh = 1.5

    ###### FIX THIS #####
    #folders = ['8_19']
    # folders = ['test']
    folders = ['GCY8Mix_Tc20_1__2']

    # RFP offets
    # Need to know how much +/- to get RFP Y dimension to 2000 and X dimension to 1000.
    # Ex: X = 1040
    #     Y = 1999
    # Therefore : x_offset = 40, y_offset = -1
    RFP_x_offset = 9
    RFP_y_offset = 48

    # GCaMP Offsets
    # The RFP and GCaMP images are also slightly offset. These values map the RFP neurite coordiantes to the cooresponding GCaMP coordinages.
    GCaMP_x_offset = 0
    GCaMP_y_offset = 0

    # voxel sizes.
    x_micron_size = 1.23
    y_micron_size = 1.23

    n_folders = len(folders)

    for i in range(n_folders):

        folder = folders[i]

        rfp_path = os.path.join(home, folder, 'Red\\')
        gcamp_path = os.path.join(home, folder, 'GCaMP\\')
        pred_path = os.path.join(home, folder, 'Predictions\\')
        out_path_tifs = os.path.join(home, folder, 'Cleaned\\')
        out_path = os.path.join(home, folder, 'CSVs\\')
        
        rfp_files = sorted([rfp_path + f for f in os.listdir(rfp_path) if f.endswith('.tif')])
        gcamp_files = sorted([gcamp_path + f for f in os.listdir(gcamp_path) if f.endswith('.tif')])
        pred_files = sorted([pred_path + f for f in os.listdir(pred_path) if f.endswith('.tif')])
        
        n_rfp_files = len(rfp_files) 
        n_gcamp_files = len(gcamp_files)
        n_pred_files = len(pred_files)

        assert(n_rfp_files == n_gcamp_files == n_pred_files)

        n_files = n_gcamp_files

        print('Folder', i+1, 'of', n_folders, '\n', folder)

        for j in range(n_files):

            st = time.time()

            rfp_str = rfp_files[j]
            gcamp_str = gcamp_files[j]
            pred_str = pred_files[j]

            rfp_vol = rfp_str.split('Red\\')[1][:-4]
            gcamp_vol = gcamp_str.split('GCaMP\\')[1][:-4]
            pred_vol = pred_str.split('Predictions\\')[1][:-4]

            print('------------------- \n')
            print(rfp_vol, gcamp_vol, pred_vol)

            rfp = tiff.imread(rfp_str)
            gcamp = tiff.imread(gcamp_str)
            pred = tiff.imread(pred_str)

            # Account for edges
            pred[-y_edge:, :] = 0
            pred[:, -x_edge:] = 0

            # # Find neuron centroid. First dilate the neuron twice to remove detected neurite around neuron.
            # pred_neuron = pred.copy()
            # pred_neuron[pred_neuron != 1] = 0

            # for z in range(neuron_dilates):
            #     pred_neuron = morphology.binary_dilation(pred_neuron)

            # # Take largest object. 
            # CCs_neuron, num_CCs_neuron = ndi.label(pred_neuron)
            # Vols_neuron, boxes_neuron = get_properties(CCs_neuron)
            # largest_vol = Vols_neuron.max()

            # for k in range(num_CCs_neuron):

            #     this_vol = Vols_neuron[k]

            #     if this_vol != largest_vol:

            #         CCs_neuron[CCs_neuron == k+1] = 0

            # neuron_centroid = ndi.center_of_mass(input=pred_neuron, labels=pred_neuron, index=[1])
            # print('Neuron centroid:', neuron_centroid)

            # pred[pred_neuron == 1] = 1


            # Isolate neurite. 
            pred[pred != 2] = 0
            pred[pred == 2] = 1

            CCs, num_CCs = ndi.label(pred)

            # get info
            Vols, Boxes = get_properties(CCs)
            num_coords = Vols.shape[0]

            # Remove debris.
            count = num_coords
            for k in range(num_coords):

                this_vol = Vols[k]

                if this_vol <= vol_thresh_rd:

                    # remove
                    CCs[CCs==k+1] = 0
                    count += -1

            print('Number of coordinates before:', num_coords)
            print('Number of coordinates after:', count)

            CCs[CCs>0] = 1

            if count > 1:

                # Try and bridge them.
                pred_temp = CCs.copy() 
                count_e_d = 0

                num_CCs_temp = num_CCs

                print('Dilating.')
                while num_CCs_temp != 1 and count_e_d < num_erosion_dilation:

                    pred_temp = morphology.binary_dilation(pred_temp)
                    CCs_temp, num_CCs_temp = ndi.label(pred_temp)
                    count_e_d += 1

                    print('--- Dilation count:', count_e_d, 'No. components:', num_CCs_temp)

                # How many times did we dilate? erode back. 
                print('Dilated', count_e_d, 'times.')
                print('Post dilation - No. components:', num_CCs_temp)

                # for i in range(count_e_d):

                #     pred_temp = morphology.binary_erosion(pred_temp)

                # Now retest.
                CCs, num_CCs = ndi.label(pred_temp)
                Vols, Boxes = get_properties(CCs)
                # print('Post Erosion - No. components:', num_CCs)

                if num_CCs > 1:

                    vols_argsort = np.argsort(Vols)

                    # Largest vol:
                    best_idx = vols_argsort[-1]
                    best_vol = Vols[best_idx]

                    # Second best:
                    second_best_idx = vols_argsort[-2]
                    second_best_vol = Vols[second_best_idx]

                    # ratio.
                    ratio = best_vol / second_best_vol

                    print('Volumes:', Vols)
                    print('Largest object volume:', best_vol)
                    print('Second largest object volume:', second_best_vol)
                    print('Ratio:', ratio)

                    if ratio > ratio_thresh:

                        # remove the rest. 
                        for i in range(num_CCs):

                            this_vol = Vols[i]

                            if this_vol != best_vol:

                                CCs[CCs==i+1] = 0

                        CCs[CCs>0] = 1
                        print('Taking largest object.')

                        CCs = img_as_ubyte(CCs)
                        _, num_CCs = ndi.label(CCs)
                        CCs = morphology.skeletonize(CCs)
                        tiff.imsave(out_path_tifs + pred_vol + '.tif', img_as_ubyte(CCs))

                    elif ratio <= ratio_thresh:

                        # Keep both - two neurites. 
                        for i in range(len(Vols)):

                            this_vol = Vols[i]

                            if this_vol < second_best_vol:

                                CCs[CCs==i+1] = 0

                        CCs[CCs>0] = 1
                        print('Taking two largest objects.')
                        CCs = img_as_ubyte(CCs)
                        _, num_CCs = ndi.label(CCs)
                        CCs = morphology.skeletonize(CCs)
                        tiff.imsave(out_path_tifs + pred_vol + '.tif', img_as_ubyte(CCs))

                elif num_CCs == 1:

                    CCs = pred_temp.copy()
                    print('Erosion - Dilation reduced to one object.')
                    CCs = img_as_ubyte(CCs)
                    CCs[CCs>0]=1
                    _, num_CCs = ndi.label(CCs)
                    CCs = morphology.skeletonize(CCs)
                    tiff.imsave(out_path_tifs + pred_vol + '.tif', img_as_ubyte(CCs))

            elif count == 1:

                print('Removing debris reduced to one object.')
                CCs = img_as_ubyte(CCs)
                _, num_CCs = ndi.label(CCs)
                CCs = morphology.skeletonize(CCs)
                tiff.imsave(out_path_tifs + pred_vol + '.tif', img_as_ubyte(CCs))

            # Now we have a skeletonized neurite with a neuron centroid. 
            # Objective now is to run breadth-first search (BFS) to move along neurite. 
            # The starting point is the point closest to the neuron centroid. 

            # Check no. neurites. 
            CCs = img_as_ubyte(CCs)
            CCs[CCs>0] = 1
            print('No. CCs:', num_CCs, 'Sum:', CCs.sum())

            # if num_CCs == 1:
                
            #     coords = np.argwhere(CCs == 1)
            #     coords = coords[::-1]
            #     num_coords = coords.shape[0]

            #     rows = []

            #     for j in range(num_coords):


            #         norm_idx = j / num_coords
            #         y, x = coords[j]
            #         this_gcamp_value = gcamp[y+y_offset, x+x_offset]

            #         this_row = [j+1, x+x_offset, y+y_offset, this_gcamp_value]
            #         rows.append(this_row)
                
            coords = np.argwhere(CCs == 1)
            coords = coords[::-1]
            num_coords = coords.shape[0]

            rows = []

            for j in range(num_coords):


                norm_idx = (j+1) / num_coords
                y, x = coords[j]
                x_um = x*x_micron_size
                y_um = y*y_micron_size

                RFP_adj_x = x+(RFP_x_offset)
                RFP_adj_y = y+(RFP_y_offset)
                #RFP_adj_x = x
                #RFP_adj_y = y

                GCaMP_adj_x = RFP_adj_x + GCaMP_x_offset
                GCaMP_adj_y = RFP_adj_y + GCaMP_y_offset

                x_um = x_micron_size*GCaMP_adj_x
                y_um = y_micron_size*GCaMP_adj_y

                this_gcamp_value = gcamp[GCaMP_adj_y, GCaMP_adj_x]
                this_rfp_value = rfp[RFP_adj_y, RFP_adj_x]

                this_ratio = this_gcamp_value / this_rfp_value

                this_row = [j+1, norm_idx, RFP_adj_x, RFP_adj_y, x_um, y_um, this_rfp_value, this_gcamp_value, this_ratio]
                rows.append(this_row)

            # Now write to CSV. 

            csv_name = os.path.join(out_path + pred_vol + '.csv')

            with open(csv_name, "w") as myfile:

                wr = csv.writer(myfile)
                wr.writerow(['Index', 'Normalized', 'X', 'Y', 'X_um', 'Y_um', 'RFP', 'GCaMP', 'GCaMP-RFP_Ratio','GCaMP_adj_x','GCaMP_adj_y'])

                for k in range(num_coords):

                    dat = rows[k]
                    wr.writerow(dat)

        print('Finsihed folder', i+1)
    print('Finished all folders!')