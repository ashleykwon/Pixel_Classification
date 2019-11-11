#import necessary packages
from PIL import Image
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt

def color_code_pixels(map, mask, element_name):
    '''
    This funciton takes as its input an element map as a numpy array and classifies pixel values into 
    num_categories categories based on how bright they are (brighter = higher pixel value)

    The function outputs a classification map in which pixels in different categories have different colors
    '''

    #color_list has 10 elements, each of which is a list with 3 positive integers
    color_list = [[75, 0, 130], [128, 0, 255], [0, 128, 255], [0, 255, 255], [0, 255, 128],[0, 255, 0], 
        [128, 255, 0],  [255, 255, 0], [255, 160, 0],[255, 0, 0]]

    row = map.shape[0]
    col = map.shape[1]


    #adds pixel values as keys and pixel locations as valus in the pxl_locations dictionary

    new_map = np.zeros((row,col))
    pxl_locations = [] 

    for i in range(row):
        for j in range(col):
            if mask[i,j] != 0:
                new_map[i,j] = map[i,j]
            else:
                continue

    for i in range(row):
        for j in range(col):
            pxl_locations.append((new_map[i,j], [i,j]))
            
    
    #classifies pixel values into 10 categories in a map that are larger than the value of the 1st quartlie
    pxl_vals = sorted([pxl_tup for pxl_tup in pxl_locations if pxl_tup[0] != 0.0 and pxl_tup[0] != 1.0], key = lambda pxl_tuple : pxl_tuple[0]) 
    first_ten_percent = pxl_vals[int(len(pxl_vals) * 0.1)][0] 
    third_quartile = pxl_vals[int(len(pxl_vals) * 0.75)][0] 
    last_ten_percent = pxl_vals[int(len(pxl_vals) * 0.9)][0] 
    print(element_name)
    print('this is the first quartile of ' + element_name + ' values ' + str(first_ten_percent))
    print('this is the third quartile of '+ element_name + ' value '+ str(third_quartile))
    print('this is the largest ten percent of '+ element_name +  ' value ' + str(last_ten_percent))
    pxl_vals = [pxl_tup for pxl_tup in pxl_vals if pxl_tup[0] > first_ten_percent]

    
    color_codes = dict()
    for i in range(10):
        color_codes[i] = pxl_vals[int(len(pxl_vals) * (i/10))][0]
    
    

    classification_map = np.zeros((row, col, 3))
    for j in range(len(pxl_vals)):
        row_idx = pxl_vals[j][1][0]
        col_idx = pxl_vals[j][1][1]

        pxl_val = pxl_vals[j][0]

        for key in list(color_codes.keys()):
            if key == len(list(color_codes.keys())) - 1:
                if pxl_val >= color_codes[9]:
                    color = color_list[key]
                    classification_map[row_idx, col_idx, 0] = color[0]
                    classification_map[row_idx, col_idx, 1] = color[1]
                    classification_map[row_idx, col_idx, 2] = color[2]

            elif 0 < key < len(list(color_codes.keys())) - 1:
                if color_codes[key - 1]<= pxl_val < color_codes[key]:
                    color = color_list[key]
                    classification_map[row_idx, col_idx, 0] = color[0]
                    classification_map[row_idx, col_idx, 1] = color[1]
                    classification_map[row_idx, col_idx, 2] = color[2]
                
                else:
                    continue
            
            else:
                if pxl_val <= color_codes[key]:
                    color = color_list[key]
                    classification_map[row_idx, col_idx, 0] = color[0]
                    classification_map[row_idx, col_idx, 1] = color[1]
                    classification_map[row_idx, col_idx, 2] = color[2]
                else:
                    continue
      

    filename = 'classified ' + element_name +  ' without the first quartile.png'
    imsave(filename, classification_map)
    return color_codes
    






if __name__ == "__main__":
    #file_path is where element maps are stored 
    file_path = '/Users/ashleykwon/Desktop/London_2019/Pixel_Classification/Element_Maps_In_Photon_Counts/'
    file_path_mask = '/Users/ashleykwon/Desktop/Pixel_Classification_Continued/'


    #load element maps and store them in a dictionary with the key as the name of the element and value as the collection of pixel 
    #values of the corresponding map
    #all maps are assumed to have the same height and width, 
    # while each of their pixel values represent the count of photons detected in that pixel location 
    #calcium_map = Image.open(file_path + "M1573_d02_decon_16bit_Ca-KA.tif")
    #chromium_map = Image.open(file_path + "M1573_d02_decon_16bit_Cr-KA.tif")
    #cobalt_map = Image.open(file_path + "M1573_d02_decon_16bit_Co-KA.tif")
    #copper_map = Image.open(file_path + "M1573_d02_decon_16bit_Cu-KA.tif")
    #iron_map = Image.open(file_path + "M1573_d02_decon_16bit_Fe-KA.tif")
    lead_l_map = np.asarray(Image.open(file_path + "M1573_d02_decon_16bit_Pb-LA.tif"))
    #lead_m_map = Image.open(file_path + "M1573_d02_decon_16bit_Pb-MA.tif")
    #manganese_map = Image.open(file_path + "M1573_d02_decon_16bit_Mn-KA.tif")
    #mercury_map = Image.open(file_path + "M1573_d02_decon_16bit_Hg-LA.tif")
    #potassium_map = Image.open(file_path +  "M1573_d02_decon_16bit_K-KA.tif")
    #tin_map = Image.open(file_path + "M1573_d02_decon_16bit_Sn_LA-Ca_KA-K_KA.tif")
    #titanium_map = Image.open(file_path + "M1573_d02_decon_16bit_Ti-KA.tif")
    #zinc_map = Image.open(file_path + "M1573_d02_decon_16bit_Zn-KA.tif")
    
    brown_map = np.asarray(Image.open(file_path_mask + "brown_calcium.png").convert('L'), dtype=np.int_)
    blue_map = np.asarray(Image.open(file_path_mask + "blue_calcium.png").convert('L'), dtype=np.int_)
    figure_map = np.asarray(Image.open(file_path_mask + "calcium_figure.png").convert('L'), dtype=np.int_)
    #element_maps = {'calcium':calcium_map, 'chromium': chromium_map,  'cobalt': cobalt_map, 'copper': copper_map, 'iron': iron_map, 'lead_l': lead_l_map, 
    #'lead_m':lead_m_map, 'mercury': mercury_map, 'titanium':titanium_map,
    #'zinc':zinc_map}
    #these are the assorted element maps based on Nathan's ppt
    # element_maps  = {'lead_l': lead_l_map}
    # for map in list(element_maps.keys()): #try binarizing the lead maps too?
    #     element_maps[map] = np.array(element_maps[map], dtype = 'float64')
    #     print(color_code_pixels(element_maps[map], map))
    #     print('\n')

    print(color_code_pixels(lead_l_map, blue_map, 'lead_l_blue'))
    print('\n')


