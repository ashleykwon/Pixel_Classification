#imports necessary packages
from PIL import Image
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import math
#import statistics
#import pyclustering as pycl
from pyclustering.cluster import kmedoids as kmed
import pickle


def normalize(data):
    '''
    This function takes as its input a two-dimensional array, which is an element map.
    Then it normalizes each entry in the map by (entry - min_entry) / (max_entry - min_entry)
    '''
    
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
           

def make_feature_matrix(data, num_eigvec):
    '''
    This function takes as its inputs 
    i) normalized nm-by-chnl data 
    ii) number of eigenvectors to choose in a list of sorted eigenvectors

    The function first calculates the covariance matrix of the data that with the size 
    n-by-n, in which n is the number of features represented in the data

    Then it finds the eigenvalues and eigenvectors of the covariance matrix and makes a sorted 
    list of (eigval, eigvec) based on eigval in descending order 

    The function returns a new feature matrix composed of num_eigvec eigenvectors, which are
    derived from the sorted list described above
    '''
    #finds the covariance matrix of data
    num_features = data.shape[1]
    features_lst = []
    for i in range(0,num_features):
        features_lst.append(data[:,i])
    cov_mat = np.cov(features_lst)
    
    #calculates eigevalues and eigenvectors of cov_mat and reshapes eigvec
    eigval, eigvec = np.linalg.eig(cov_mat)

    #initiate and fill eig_pairs list with (eigval, eigvec tuples)
    eig_pairs = [(np.abs(eigval[k]), eigvec[:,k]) for k in range(len(eigval))]   
    
    #sorts eig_pairs in descending order based on the value of eigval in each tuple
    eig_pairs.sort(key = lambda x: x[0], reverse = True) 

    #chooses num_eigvec eigenvectors from eig_pairs and combines them into a matrix
    chosen_eigvecs = []
    for i in range(num_eigvec):
        chosen_eigvecs.append(eig_pairs[i][1])
    new_feature_matrix = np.hstack(tuple(chosen_eigvecs))
    new_feature_matrix = np.reshape(new_feature_matrix, (num_features,num_eigvec))


    return new_feature_matrix


def modify_data(data, new_feat_matrix):
    '''
    This function takes as its inputs 
    i) normalized data
    ii) a feature matrix composed of eigenvectors chosen in the function, make_feature_matrix

    Then it reduces the dimension of the data by the number of eigvecs 
    and returns the data with reduced dimension 
    '''
    ret = data.dot(new_feat_matrix)
    return ret

def PCA(data2D, num_dimensions):
    '''
    This function reduces the number of dimensions in data2D to num_dimensions 

    to be able to plot the result, num_dimensions should be 3
    '''
    #reduces the dimension of the data to num_dimensions
    data2D_PCA = normalize(data2D)
    feat_matrix_data2D_PCA = make_feature_matrix(data2D_PCA, num_dimensions)
    data2D_PCA = modify_data(data2D_PCA, feat_matrix_data2D_PCA)
    return data2D_PCA

def run_kmedoids(element_maps_with_binary_lead_l, binarized_lead_l_map, num_clusters, num_clusters_in_a_cluster):
    '''
    This function creates 2 clusters based on a binarized lead_l map, one with areas where lead is present and the other with
    area where lead is not present (under the threshold level)
    '''
    row = binarized_lead_l_map.shape[0]
    col = binarized_lead_l_map.shape[1]

    combined_maps = np.zeros((row,col))

   
    for map in list(element_maps_with_binary_lead_l.keys()):
        combined_maps = np.dstack((combined_maps, element_maps_with_binary_lead_l[map]))
    
    unnormalized_data = combined_maps[:, :, 1:]
    num_chnl = unnormalized_data.shape[2]
    
    #this part normalizes each map in the patch
    normalized_data = np.zeros((row, col, num_chnl)) #each patch is normalized with different minimum and maximum values
    for i in range(num_chnl):
        normalized_data[:, :, i]  =  normalize(unnormalized_data[:, :, i])
    
    #reshapes the data to run kmeans
    data2D = np.reshape(normalized_data, (row*col, num_chnl))
    #data2D_PCA = PCA(data2D, 1) #reduces  data2D to have 1 dimension only so that it can be given to the kmedoids function 
    #data 2d size : 1345410,11
    initial_index_medoids = [1,30000]
    kmed_round1 = kmed.kmedoids(data2D, initial_index_medoids)
    kmed_round1.process()
    result_1 = kmed_round1.get_clusters()
    classified_result_1 = np.full((row*col), 255)
    for i in range(len(result_1)):
        for j in range(len(result_1[i])):
            classified_result_1[result_1[i][j]] = i
    classified_result_1 = np.reshape(classified_result_1, (row,col))

    cluster_dict = dict()
    pixel_location_dict = dict()

    # stores in each of the two dictionaries above classification information for each pixel in the first
    # round of clustering and the location of the pixel
    for i in range(classified_result_1.shape[0]):
        for j in range(classified_result_1.shape[1]):
            if classified_result_1[i,j] not in cluster_dict:
                cluster_dict[classified_result_1[i,j]] = [[] for num in range(num_chnl)]
                pixel_location_dict[classified_result_1[i,j]] = []
            pixel_location_dict[classified_result_1[i,j]].append([i,j])
            for k in range(num_chnl):
                cluster_dict[classified_result_1[i,j]][k].append(normalized_data[i,j,k])

    #runs the second round of classification on each of the clusters formed from the first classification 
    for cluster in list(cluster_dict.keys()):
        cluster_dict[cluster] = np.array(cluster_dict[cluster], dtype = 'float32')
        cluster_dict[cluster] = np.transpose(cluster_dict[cluster]) #each value for a cluster is a row*col , num_chnl
        clusters_in_a_cluster = cluster_clusters(cluster_dict[cluster], cluster, num_clusters_in_a_cluster)
        cluster_dict[cluster] =  clusters_in_a_cluster
    
    #builds the classification map based on the second classification results
    for cluster in list(cluster_dict.keys()):
        classification_map = np.full((row, col), 255)
        for i in range(len(cluster_dict[cluster])):
            idx_pair = pixel_location_dict[cluster][i] 
            row_idx = idx_pair[0]
            col_idx = idx_pair[1]
            classification_map[row_idx, col_idx] = cluster_dict[cluster][i]
        build_map(classification_map, cluster, num_clusters_in_a_cluster)
        
    return result_1

def cluster_clusters(data_from_cluster, cluster_id, num_clusters_in_a_cluster):
    '''
    This function takes as its input a numpy array of elements that ended up in one cluster.

    The cluster is a numpy array with the dimension of that is equal to the number of element maps originally given to the
    algorithm. All values in the cluster are already normalized

    Then, the function runs the kmeans classification in those cluster elements. 
    '''

    initial_index_medoids = []

    for i in range(num_clusters_in_a_cluster):
        initial_index_medoids.append(i*300)

    row = data_from_cluster.shape[0]
    col = data_from_cluster.shape[1]
    
    kmed_round2 = kmed.kmedoids(data_from_cluster, initial_index_medoids)
    kmed_round2.process()
    result_2 = kmed_round2.get_clusters()
    ret_size = len(result_2[0]) + len(result_2[1]) + len(result_2[2])
    classified_result_2 = np.full(ret_size, 255)
    for i in range(len(result_2)):
        for j in range(len(result_2[i])):
            classified_result_2[result_2[i][j]] = i

    center_list = kmed_round2.get_medoids()
    for i in range(len(center_list)):
        center_list[i] = data_from_cluster[center_list[i],:]
    element_names = ['calcium', 'cobalt', 'copper', 'iron', 'manganese', 'mercury', 'potassium', 'tin', 'zinc', 'combined_cr_and_ti', 'lead_l']
    print_centers(center_list, element_names, cluster_id)
    #classified_result_2 = np.reshape(classified_result_2, (row,col))


    return classified_result_2


def build_map(result_to_return, num_clusters, num_clusters_in_a_cluster):
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
                colored_map[i,j,1] = 127
            elif result_to_return[i,j] == 5:
                colored_map[i,j,0] = 255
                colored_map[i,j,2] = 255
            else:
                continue
    filename = 'kmed result for cluster ' + str(num_clusters) + ' with '+ str(num_clusters_in_a_cluster) + ' sub-clusters.png'
    imsave(filename,colored_map)
    print(filename + 'saved')
    return colored_map





def print_centers(center_list, element_names, cluster_id):
    '''
    This function takes as its input a list of lists, each of whose elements are center values in a cluster and
    a list of names of element maps used in running the kmeans algorithm
    '''
    for i in range(len(center_list)):
        lst_to_print = center_list[i]
        lst_to_save = dict()
        for j in range(len(lst_to_print)):
            print(element_names[j] + ' center in cluster '+ str(cluster_id) + ' sub-cluster '+ str(i) + ' : ' + str(lst_to_print[j]))
            lst_to_save[element_names[i]] = center_list[i]
            file = open('saved_sub_cluster'+str(i), 'wb')
            pickle.dump(lst_to_save, file)
        print('\n')
    

    return 'done'




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
    
    
    #creats the binary map 
    for i in range(row):
        for j in range(col):
            if map[i,j] >= thresholds[element_name]:
                map[i,j] = max_val
            else:
                map[i,j] = 0.0
    print(element_name + ' binarized')
    return map



def run_kmedoids_and_plot(maps_lead_separated, num_runs):
    '''
    This function takes in as its input a dictionary that has the name of an element as its key
    and the corresponding map data of the element as its value.
    inertia_ is "sum of squared distances of samples to their closest cluster center."
    '''
    row = maps_lead_separated.shape[0]
    col = maps_lead_separated.shape[1]
    num_chnl = maps_lead_separated.shape[2]
   
    # normalizes the lead map of choice
    normalized_lead_separated = np.zeros((row, col, num_chnl)) #each patch is normalized with different maximum values
    for i in range(num_chnl):
        normalized_lead_separated[:, :, i]  =  normalize(maps_lead_separated[:, :, i])
    data2D_lead_separated = np.reshape(normalized_lead_separated, (row*col, num_chnl))
    
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



if __name__ == "__main__":
    #CROPPED MAPS IN 8-BIT
    # file_path = '/Users/ashleykwon/Desktop/Pixel_Classification_Continued/'

    # calcium_map = Image.open(file_path + "cropped_calcium.png")
    # chromium_map = Image.open(file_path + "cropped_chromium.png")
    # cobalt_map = Image.open(file_path + "cropped_cobalt.png")
    # copper_map = Image.open(file_path + "cropped_copper.png")
    # iron_map = Image.open(file_path + "cropped_iron.png")
    # lead_l_map = Image.open(file_path + "cropped_lead_l.png")
    # #lead_m_map = Image.open(file_path + "M1573_d02_decon_16bit_Pb-MA.tif")
    # manganese_map = Image.open(file_path + "cropped_manganese.png")
    # mercury_map = Image.open(file_path + "cropped_mercury.png")
    # potassium_map = Image.open(file_path +  "cropped_potassium.png")
    # tin_map = Image.open(file_path + "cropped_tin.png")
    # titanium_map = Image.open(file_path + "cropped_titanium.png")
    # zinc_map = Image.open(file_path + "cropped_zinc.png")
    

    #ORIGINAL MAPS THAT CATHERINE SENT ME
    #file_path = '/Users/ashleykwon/Desktop/London_2019/Pixel_Classification/Element_Maps_In_Photon_Counts/'
    file_path = '/home/ugrad/akwon216/Desktop/'
    calcium_map = Image.open(file_path + "M1573_d02_decon_16bit_Ca-KA.tif")
    chromium_map = Image.open(file_path + "M1573_d02_decon_16bit_Cr-KA.tif")
    cobalt_map = Image.open(file_path + "M1573_d02_decon_16bit_Co-KA.tif")
    copper_map = Image.open(file_path + "M1573_d02_decon_16bit_Cu-KA.tif")
    iron_map = Image.open(file_path + "M1573_d02_decon_16bit_Fe-KA.tif")
    lead_l_map = Image.open(file_path + "M1573_d02_decon_16bit_Pb-LA.tif")
    #lead_m_map = Image.open(file_path + "M1573_d02_decon_16bit_Pb-MA.tif")
    manganese_map = Image.open(file_path + "M1573_d02_decon_16bit_Mn-KA.tif")
    mercury_map = Image.open(file_path + "M1573_d02_decon_16bit_Hg-LA.tif")
    potassium_map = Image.open(file_path +  "M1573_d02_decon_16bit_K-KA.tif")
    tin_map = Image.open(file_path + "M1573_d02_decon_16bit_Sn_LA-Ca_KA-K_KA.tif")
    titanium_map = Image.open(file_path + "M1573_d02_decon_16bit_Ti-KA.tif")
    zinc_map = Image.open(file_path + "M1573_d02_decon_16bit_Zn-KA.tif")
    element_maps  = {'calcium':calcium_map, 'cobalt': cobalt_map, 
    'copper': copper_map, 'iron': iron_map, 'manganese': manganese_map, 'mercury':mercury_map, 'tin': tin_map,
    'potassium': potassium_map, 'tin': tin_map, 'zinc': zinc_map}
    
    NUM_CLUSTERS = 2
    NUM_CLUSTERS_IN_A_CLUSTER = 3

    element_maps = {'calcium':calcium_map, 'cobalt': cobalt_map, 'copper': copper_map, 'iron': iron_map, 'manganese': manganese_map, 'mercury':mercury_map, 'potassium': potassium_map, 'tin': tin_map, 'zinc': zinc_map}
    
    

    #converts each map in element_maps to a numpy array
    for map in list(element_maps.keys()): 
        element_maps[map] = np.array(element_maps[map], dtype = 'float32')
        #the following line is necessary when the code is run on the 8-bit images.
        #element_maps[map] = np.reshape(element_maps[map][:, :, 0], (element_maps[map].shape[0], element_maps[map].shape[1]))
    
    chromium_map = np.array(chromium_map, dtype = 'float32')
    #the following line is necessary when the code is run on the 8-bit images.
    #chromium_map = np.reshape(chromium_map[:, :, 0], (chromium_map.shape[0], chromium_map.shape[1]))

    titanium_map = np.array(titanium_map, dtype  = 'float32')
    #the following line is necessary when the code is run on the 8-bit images.
    #titanium_map = np.reshape(titanium_map[:, :, 0], (titanium_map.shape[0], titanium_map.shape[1]))

    lead_l_map = np.array(lead_l_map, dtype  = 'float32')
    #the following line is necessary when the code is run on the 8-bit images
    #lead_l_map = np.reshape(lead_l_map[:, :, 0], (titanium_map.shape[0], titanium_map.shape[1]))


    binarized_cr_map = binarize(chromium_map, 'chromium')
    binarized_ti_map = binarize(titanium_map, 'titanium')


    #make a combined map of chromium and titanium
    combined_cr_and_ti_map = np.zeros((binarized_cr_map.shape[0],binarized_cr_map.shape[1]))
    for i in range(binarized_cr_map.shape[0]):
        for j in range(binarized_cr_map.shape[1]):
            if binarized_cr_map[i,j] != 0.0 or binarized_ti_map[i,j] != 0.0:
                combined_cr_and_ti_map[i,j] = 255
    element_maps['combined_cr_and_ti'] = combined_cr_and_ti_map

 

    binarized_lead_l_map = binarize(lead_l_map, 'lead_l')
    element_maps['lead_l'] =  binarized_lead_l_map
    new_element_maps = remove_ti_and_cr(element_maps, binarized_ti_map, binarized_cr_map)

    
    res = run_kmedoids(new_element_maps, element_maps['lead_l'], NUM_CLUSTERS, NUM_CLUSTERS_IN_A_CLUSTER)
    #res_plot = run_kmedoids_and_plot(maps_lead_absent, 15)
    
