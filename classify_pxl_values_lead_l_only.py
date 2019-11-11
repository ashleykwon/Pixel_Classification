#import necessary packages
from PIL import Image
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt

def color_code_pixels(map, element_name):
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
    pxl_locations = [] 
    for i in range(row):
        for j in range(col):
            pxl_locations.append((map[i,j], [i,j]))
    
    
    pxl_vals = sorted([pxl_tup for pxl_tup in pxl_locations if pxl_tup[0] != 0.0], key = lambda pxl_tuple : pxl_tuple[0]) 
    first_quartile = pxl_vals[int(len(pxl_vals) * 0.25)][0] 
    third_quartile = pxl_vals[int(len(pxl_vals) * 0.75)][0] 
    last_ten_percent = pxl_vals[int(len(pxl_vals) * 0.9)][0] 
    print(element_name)
    print('this is the first quartile of ' + element_name + ' values ' + str(first_quartile))
    print('this is the third quartile of '+ element_name + ' value '+ str(third_quartile))
    print('this is the largest ten percent of '+ element_name +  ' value ' + str(last_ten_percent))
    #if element_name  == 'tin' or element_name == 'zinc':
    #    pxl_vals = [pxl_tup for pxl_tup in pxl_vals if pxl_tup[0] > 9.0]
    #else:
    pxl_vals = [pxl_tup for pxl_tup in pxl_vals]

    
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
      

    filename = 'classified ' + element_name +  ' with all quartiles.png'
    imsave(filename, classification_map)
    return color_codes
    






if __name__ == "__main__":
    #file_path is where element maps are stored 
    file_path = '/Users/ashleykwon/Desktop/London_2019/Pixel_Classification/Element_Maps_In_Photon_Counts/'


    #load element maps and store them in a dictionary with the key as the name of the element and value as the collection of pixel 
    #values of the corresponding map
    #all maps are assumed to have the same height and width, 
    # while each of their pixel values represent the count of photons detected in that pixel location 
    #calcium_map = Image.open(file_path + "M1573_d02_decon_16bit_Ca-KA.tif")
    #chromium_map = Image.open(file_path + "M1573_d02_decon_16bit_Cr-KA.tif")
    #cobalt_map = Image.open(file_path + "M1573_d02_decon_16bit_Co-KA.tif")
    #copper_map = Image.open(file_path + "M1573_d02_decon_16bit_Cu-KA.tif")
    #iron_map = Image.open(file_path + "M1573_d02_decon_16bit_Fe-KA.tif")
    lead_l_map = Image.open(file_path + "M1573_d02_decon_16bit_Pb-LA.tif")
    #lead_m_map = Image.open(file_path + "M1573_d02_decon_16bit_Pb-MA.tif")
    #manganese_map = Image.open(file_path + "M1573_d02_decon_16bit_Mn-KA.tif")
    #mercury_map = Image.open(file_path + "M1573_d02_decon_16bit_Hg-LA.tif")
    #potassium_map = Image.open(file_path +  "M1573_d02_decon_16bit_K-KA.tif")
    #tin_map = Image.open(file_path + "M1573_d02_decon_16bit_Sn_LA-Ca_KA-K_KA.tif")
    #titanium_map = Image.open(file_path + "M1573_d02_decon_16bit_Ti-KA.tif")
    #zinc_map = Image.open(file_path + "M1573_d02_decon_16bit_Zn-KA.tif")
    

    #element_maps = {'calcium':calcium_map, 'chromium': chromium_map,  'cobalt': cobalt_map, 'copper': copper_map, 'iron': iron_map, 'lead_l': lead_l_map, 
    #'lead_m':lead_m_map, 'mercury': mercury_map, 'titanium':titanium_map,
    #'zinc':zinc_map}
    
    #these are the assorted element maps based on Nathan's ppt
    element_maps  = {'lead_l': lead_l_map, 'lead_m': lead_m_map}
    for map in list(element_maps.keys()): #try binarizing the lead maps too?
        element_maps[map] = np.array(element_maps[map], dtype = 'float64')
        print(color_code_pixels(element_maps[map], map))
        print('\n')

