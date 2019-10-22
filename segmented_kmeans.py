#imports necessary packages
from PIL import Image
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import math
import statistics


def normalize(data):
    '''
    This function takes as its input a two-dimensional array, which is an element map.
    Then it normalizes each entry in the map by (entry - min_entry) / (max_entry - min_entry)
    '''
    #return data/np.max(data)
    max_val = data[0,0]
    normalized = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] >= max_val:
                max_val = data[i,j]
    if max_val == 0:
        return data
    else: 
        for i in range(normalized.shape[0]):
            for j in range(normalized.shape[1]):
                normalized[i,j] = float(data[i,j] / max_val)
        return normalized 



def remove_ti_and_cr(element_maps, binarized_ti_map, binarized_cr_map):
    '''
    This function takes as input a dictionary of element maps, composed of elements other than titanium, chromium,
    and lead and changes the values on those maps where titanium and/or chromium are detected to 0

    The function outputs the updated dictionary
    '''
    row = binarized_ti_map.shape[0]
    col = binarized_ti_map.shape[1]
    for map in list(element_maps.keys()):
        for i in range(row):
            for j in range(col):
                if binarized_ti_map[i,j] != 0.0 or binarized_cr_map[i,j] != 0.0:
                    element_maps[map][i,j] = 0.0
                else:
                    continue
       
    return element_maps

def get_ti_and_cr(element_maps, binarized_ti_map, binarized_cr_map):
    '''
    This function takes as input a dictionary of element maps, composed of elements other than titanium, chromium,
    and lead and changes the values on those maps where titanium and/or chromium are detected to 0

    The function outputs the updated dictionary
    '''
    row = binarized_ti_map.shape[0]
    col = binarized_ti_map.shape[1]
    for map in list(element_maps.keys()):
        for i in range(row):
            for j in range(col):
                if binarized_ti_map[i,j] == 0.0 or binarized_cr_map[i,j] == 0.0:
                    element_maps[map][i,j] = 0.0
                else:
                    continue
       
    return element_maps
           


def separate_lead_l(element_maps, binarized_lead_l_map, lead_l_max):
    '''
    This function creates 2 clusters based on a binarized lead_l map, one with areas where lead is present and the other with
    area where lead is not present (under the threshold level)
    '''
    row = binarized_lead_l_map.shape[0]
    col = binarized_lead_l_map.shape[1]

    lead_absent = np.zeros((row,col))
    lead_present = np.zeros((row,col))

    for i in range(row):
        for j in range(col):
            if binarized_lead_l_map[i,j] == 0.0:
                lead_absent[i,j] = lead_l_max
            else:
                lead_present[i,j] = lead_l_max
    
    maps_lead_absent = np.zeros((row,col))
    maps_lead_present = np.zeros((row,col))            
    
    for map in list(element_maps.keys()):
        updated_map_lead_present = np.zeros((element_maps[map].shape[0], element_maps[map].shape[1]))
        updated_map_lead_absent = np.zeros((element_maps[map].shape[0], element_maps[map].shape[1]))
        for k in range(row):
            for r in range(col):
                if lead_present[k,r] != 0.0:
                    updated_map_lead_present[k,r] = element_maps[map][k,r]
                else:
                    updated_map_lead_absent[k,r] = element_maps[map][k,r]
        maps_lead_present = np.dstack((maps_lead_present, updated_map_lead_present))
        maps_lead_absent = np.dstack((maps_lead_absent, updated_map_lead_absent))

    return (maps_lead_present[:, :, 1:], maps_lead_absent[:, :, 1:], lead_present, lead_absent)



def run_kmeans(maps, num_clusters, mask_out): #give this function the combined ti, cr maps, and segment -> replace the mask_out parameter with this combination 
    '''
    This function takes in as its input combined element maps with the dimension that is 
    equal to the number of element maps given to the algorithm

    It returns the result of running the kmeans algorithm on a segment where lead is present and on the other segment where lead
    is not present in a tuple. By calling the build_map function, it also constructs and saves in PNG files
    classification results of those two segments from which parts where titanium and/or chromium were detected were removed 
    '''
    num_chnl = maps.shape[0]
    row = maps.shape[1]
    col = maps.shape[2]
    

    #the following list should be modified based on what element maps, besides binarized chromium, titanium, and lead_l maps 
    #were used in the classification. The list is only used to print center values in each cluster
    element_names = ['calcium', 'chromium', 'cobalt', 'iron', 'lead_l', 'manganese', 'potassium', 'tin', 'titanium']
    mask_out = mask_out.flatten() #sorts out zero pixels like those from ti and cr maps 
    # normalizes each element map
    # normalized_maps = np.zeros((row, col, num_chnl)) #each patch is normalized with different maximum values
    data2D = np.zeros((row*col, num_chnl)) #each patch is normalized with different maximum values
    data2D_masked = np.zeros((int(sum(mask_out)), num_chnl)) #each patch is normalized with different maximum values
    for i in range(num_chnl):
        data2D[:,i] = maps[i, :, :].flatten()
        data2D_masked[:,i] = data2D[mask_out,i] 


    #runs clustering in the segment where lead is present
    kmeans_lead_present  = KMeans(n_clusters = num_clusters, random_state = 0).fit(data2D)
    result = np.zeros(data2D.shape[0]) + 10
    result[mask_out] = kmeans_lead_present.predict(data2D_masked)
    result = np.reshape(result, (row, col))
    build_map(result, num_clusters)
    centers = kmeans_lead_present.cluster_centers_
    print_centers(centers, element_names)


    return 'done'


def print_centers(center_list, element_name):
    '''
    This function takes as its input a list of lists, each of whose elements are center values in a cluster,
    a list of names of element maps used in running the kmeans algorithm, 
    and a boolean value showing whether lead was present in those maps or not
    '''
    
    for i in range(len(center_list)):
        lst_to_print = center_list[i]
        for j in range(len(lst_to_print)):
            print(element_name[j] + ' center in cluster ' + str(i) + ' : ' + str(lst_to_print[j]))
        print('\n')
    return 'done'



def build_map(result_to_return, num_clusters):
    '''
    This function takes as its input the result of clustering and a boolean value representing whether lead is
    present in the maps considered in result_to_return
    
    The map turns result_to_returninto a color-coded classification map and outputs the classification map.
    '''
    row = result_to_return.shape[0]
    col = result_to_return.shape[1]
    colored_map = np.zeros((row, col, 3))
    for i in range(row):
        for j in range(col):
            if result_to_return[i,j] == 0:
                colored_map[i,j,0] = 255
            elif result_to_return[i,j] == 1:
                colored_map[i,j,1] = 255
            elif result_to_return[i,j] == 2:
                colored_map[i,j,2] = 255
            elif result_to_return[i,j] == 3:
                colored_map[i,j,0] = 127
            elif result_to_return[i,j] == 4:
                colored_map[i,j,0] = 127
                colored_map[i,j,1] = 255
            elif result_to_return[i,j] == 5:
                colored_map[i,j,0] = 255
                colored_map[i,j,2] = 255
            else:
                continue
    filename = 'round 1 result for blue segment with ' + str(num_clusters)  + ' clusters and non-binary ti and cr.png'
    imsave(filename, colored_map)
    
    return colored_map




def binarize(map, element_name):
    '''
    In this function, map is a numpy array that is an element map. 

    element_name is the name of the element that the map represents.

    The function takes the maximum pixel value in the map and turns 
    all pixels whose values are higher than the corresponding threshold value defined in the dectionary to the maximum value
    so that when the map is normalized, its pixels only have the values of 0.0 or 1.0
    '''
    thresholds = {'chromium': 5.0, 'titanium': 9.0, 'lead_l' : 166.0}
    row = map.shape[0]
    col = map.shape[1]
    
    #calculates max_val in the given map
    max_val = map[0,0]
    for i in range(row):
        for j in range(col):
            if map[i,j] >= max_val:
                max_val = map[i,j]
    #map map[map>threshold] = 0 / map[map <= threshold]
    
    #creats the binary map 
    for i in range(row):
        for j in range(col):
            if map[i,j] >= thresholds[element_name]:
                map[i,j] = max_val
            else:
                map[i,j] = 0.0


    #filename = 'binarized_' + element_name + '_map_with_thresh_' + str(thresholds[element_name])+'.png'
    #imsave(filename, map)
    return map

def run_kmeans_and_plot(maps_lead_separated, num_runs):
    '''
    This function takes in as its input a dictionary that has the name of an element as its key
    and the corresponding map data of the element as its value.
    inertia_ is "sum of squared distances of samples to their closest cluster center."
    '''
    num_chnl = maps_lead_separated.shape[0]
    row = maps_lead_separated.shape[1]
    col = maps_lead_separated.shape[2]
    
   
    data2D_lead_separated = np.reshape(maps_lead_separated, (row*col, num_chnl))
    
    #runs the kmeans algorithm until the number of clusters reaches the given maximum number of clusters value and plots
    #in a 2D space number of clusters vs. error
    num_runs = num_runs
    num_clusters = []
    error_vals = [] 
    for i in range(1, num_runs + 1):
        kmeans = KMeans(n_clusters = i).fit(data2D_lead_separated)
        num_clusters.append(i)
        default_error = kmeans.inertia_
        error_vals.append(float(default_error / (row*col*num_chnl)))
    plt.plot(num_clusters, error_vals)
    plt.show()

    return 'done'

def mask(eight_bit_img, sixteen_bit_img):
    row = eight_bit_img.shape[0]
    col = eight_bit_img.shape[1]
    for i in range(row):
        for j in range(col):
            if eight_bit_img[i,j] == 0.0:
                sixteen_bit_img[i,j] = 0.0
            else:
                continue
    print('masking done')
    return sixteen_bit_img
    


    

if __name__ == "__main__":
    file_path_mask = '/Users/ashleykwon/Desktop/Pixel_Classification_Continued/'
    file_path = '/Users/ashleykwon/Desktop/London_2019/Pixel_Classification/Element_Maps_In_Photon_Counts/'


    #load element maps and store them in a dictionary with the key as the name of the element and value as the collection of pixel 
    #values of the corresponding map
    #all maps are assumed to have the same height and width, 
    # while each of their pixel values represent the count of photons detected in that pixel location 
    calcium_map = np.asarray(Image.open(file_path + "M1573_d02_decon_16bit_Ca-KA.tif"), dtype=np.double)
    chromium_map = np.asarray(Image.open(file_path + "M1573_d02_decon_16bit_Cr-KA.tif"), dtype=np.double)
    cobalt_map = np.asarray(Image.open(file_path + "M1573_d02_decon_16bit_Co-KA.tif"), dtype=np.double)
    iron_map = np.asarray(Image.open(file_path + "M1573_d02_decon_16bit_Fe-KA.tif"), dtype=np.double)
    lead_l_map = np.asarray(Image.open(file_path + "M1573_d02_decon_16bit_Pb-LA.tif"), dtype=np.double)
    manganese_map = np.asarray(Image.open(file_path + "M1573_d02_decon_16bit_Mn-KA.tif"), dtype=np.double)
    potassium_map = np.asarray(Image.open(file_path + "M1573_d02_decon_16bit_Hg-LA.tif"), dtype=np.double)
    tin_map = np.asarray(Image.open(file_path + "M1573_d02_decon_16bit_Sn_LA-Ca_KA-K_KA.tif"), dtype=np.double)
    titanium_map = np.asarray(Image.open(file_path + "M1573_d02_decon_16bit_Ti-KA.tif"), dtype=np.double)
    

    element_maps = {'calcium':calcium_map, 'cobalt': cobalt_map, 'iron': iron_map, 'lead_l': lead_l_map, 'manganese': manganese_map, 'potassium': potassium_map, 'tin':tin_map}
    

    NUM_CLUSTERS = 4
    
    element_maps_round1 = {'calcium':calcium_map, 'chromium': chromium_map, 'cobalt': cobalt_map, 'iron': iron_map, 'lead_l': lead_l_map, 'manganese': manganese_map, 'potassium': potassium_map, 'tin':tin_map, 'titanium':titanium_map}
    element_maps_round2 = {'calcium':calcium_map, 'cobalt': cobalt_map, 'iron': iron_map, 'lead_l': lead_l_map, 'manganese': manganese_map, 'potassium': potassium_map, 'tin':tin_map}
        
    binarized_cr_map = binarize(np.array(chromium_map), 'chromium')
    binarized_ti_map = binarize(np.array(titanium_map), 'titanium')


    #element_maps_round1 = {'calcium':calcium_map, 'chromium': binarized_cr_map, 'cobalt': cobalt_map, 'iron': iron_map, 'lead_l': lead_l_map, 'manganese': manganese_map, 'potassium': potassium_map, 'tin':tin_map, 'titanium':binarized_ti_map}

    #this is where I choose the segment to mask and crop maps accordingly 
    brown_map = np.asarray(Image.open(file_path_mask + "brown_calcium.png").convert('L'), dtype=np.int_)
    blue_map = np.asarray(Image.open(file_path_mask + "blue_calcium.png").convert('L'), dtype=np.int_)
    figure_map = np.asarray(Image.open(file_path_mask + "calcium_figure.png").convert('L'), dtype=np.int_)

   

    new_map = np.zeros((binarized_cr_map.shape[0], binarized_cr_map.shape[1]))
    for i in range(binarized_cr_map.shape[0]):
        for j in range(binarized_cr_map.shape[1]):
            if binarized_cr_map[i,j] != 0:
                new_map[i,j] = binarized_cr_map[i,j]
            if binarized_ti_map[i,j] != 0:
                new_map[i,j] = binarized_ti_map[i,j]
            else:
                continue
    EIGHT_BIT_IMG = blue_map != 0  #this is where I modify the mask
    binzrized_ti_cr = new_map != 0 
    
    combined_list = []
    for map in list(element_maps_round1.keys()): #change element_maps to element_maps_round1 or element_maps_round2
        #element_maps_round1[map] = mask(EIGHT_BIT_IMG, element_maps[map])
        element_maps_round1[map] = normalize(element_maps_round1[map])
        combined_list.append(element_maps_round1[map])
    combined_array =np.array(combined_list)


    #maps_lead_present = separate_lead_l(element_maps, binarized_lead_l_map, lead_l_max)[0]
    #maps_lead_absent = separate_lead_l(element_maps, binarized_lead_l_map, lead_l_max)[1]
    res = run_kmeans(combined_array, NUM_CLUSTERS, EIGHT_BIT_IMG)
    #plotted = run_kmeans_and_plot(combined_array, 10)
   

    
