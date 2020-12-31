import numpy as np
import pywt
from PIL import Image
import scipy
import scipy.optimize
import pickle
import time
from matplotlib import pyplot as pt

# CONSTANTS
###########
WAVELET_TYPE = 'db1' # 'haar', 'db', 'wei'
if WAVELET_TYPE=='haar':
    NUM_MAT = 4
elif WAVELET_TYPE=='db1':
    NUM_MAT = 4
elif WAVELET_TYPE=='wei':
    NUM_MAT = 7

IM_DIM = np.array([580, 1716])
MAT_DIM = (IM_DIM/2 ).astype(np.int)

ERR_C = []
ERR_A = []
ERR_K_MEANS = []
ERR_PI = []
ERR_C_LAST = [-1]
ERR_A_LAST = [-1]
ERR_K_MEANS_LAST = [-1]
ERR_PI_LAST = [-1]




# Functions
###########

def waveletTransform(originalImage): #this gives vector C
    approx, details = pywt.dwt2(originalImage, WAVELET_TYPE) #will only give the first level
    approx = np.array(approx)
    details = np.array(details)
    #concatenate all coefficients in approx and details into a vector 
    coefficients = approx.flatten()
    coefficients = np.concatenate((coefficients, details[0].flatten()))
    coefficients = np.concatenate((coefficients, details[1].flatten()))
    coefficients = np.concatenate((coefficients, details[2].flatten()))
    return coefficients #has the same number of pixels as the total number of pixels in the given elemental map
    
 
def findA(coeffs, elementsPerPigment):
    A = np.linalg.lstsq(elementsPerPigment.T, coeffs) 
    return A[0] 


def cost(arr, F, a, b, c, d, numDynamicPigments, numElements, coeffs, HSCube, elementsPerPigment):
    arr  = np.reshape(arr, (numDynamicPigments,numElements))
    elementsPerPigment[:numDynamicPigments, :] = arr
    newA = findA(coeffs, elementsPerPigment)  
    numPixels = HSCube.shape[0]*HSCube.shape[1]

    #add another constraint that entries in newA should range from 0 to 1?
    probabilities = undoWaveTrans(newA, IM_DIM[0], IM_DIM[1], numPigments) #should range from 0 to 1

    TwoDimHSCube = np.reshape(HSCube, (numPixels, HSCube.shape[2])) 
    #probabilities in each row of probabilityA is the probability that pixel at the i-th row of A is likely to be in the k-th cluster
    #for each image in probabilities, find spectral average, spectral distances from the average

    kmeansError = 0
    for pig in range(numPigments):
        weightedAvg = (TwoDimHSCube.T.dot(probabilities[pig:pig+1,:].T))/TwoDimHSCube.shape[0]
        kmeansError += np.linalg.norm(TwoDimHSCube.T - weightedAvg, axis = 0).dot(probabilities[pig,:].T)
    kmeansError = kmeansError / numPigments
    ERR_C_LAST[0] = d * np.linalg.norm(coeffs.T - newA.T@elementsPerPigment)**2
    ERR_A_LAST[0] = a * specialNorm(newA)
    ERR_K_MEANS_LAST[0] = b * kmeansError
    ERR_PI_LAST[0] = c * np.linalg.norm(elementsPerPigment.flatten(), ord = 1/2)
    return ERR_C_LAST[0] + ERR_A_LAST[0] + ERR_K_MEANS_LAST[0] + ERR_PI_LAST[0]


def undoWaveTrans(newA, numRows, numCols, numPigments):
    undoTransform = np.zeros((numPigments, numRows*numCols))
    #take a row from A, and return a 2-dimensional image
    #should be a 3-dimensional array with the size (n , m , numPigments)
    #A is of size (n*m, numPigments)
    for pig_i in range(newA.shape[0]):
        pig_coeffs = newA[pig_i, :]
        undoTransform[pig_i, :] = DWTInvert(pig_coeffs , NUM_MAT, MAT_DIM, WAVELET_TYPE).flatten()
        undoTransform[pig_i, :] = NormalizeTo01(undoTransform[pig_i,:])
    undoTransform = undoTransform/np.sum(undoTransform, axis=0)
    return undoTransform #is a 2D array in which each column corresponds to one image representing a pigment

def NormalizeTo01(im):
    im = im - np.min(im)
    im = im/np.max(im)
    return im

def specialNorm(A):
    #A is of size (J * K)
    return np.sum(np.sum(np.abs(A), axis=0))**2


def DWTInvertMany(dwt_coeffs_flat, num_mat, mat_dim, wavelet_base):
    im_stack = []
    for flat_coeff in dwt_coeffs_flat:
        im_stack.append(DWTInvert(flat_coeff, num_mat, mat_dim, wavelet_base).flatten())
    return np.array(im_stack)


def DWTInvert(dwt_coeffs_flat, num_mat, mat_dim, wavelet_base):
    num_pixels = len(dwt_coeffs_flat)
    dwt_coeffs = dwt_coeffs_flat.reshape([num_mat, np.int(num_pixels/num_mat)])
    dwt_coeffs = dwt_coeffs.reshape([num_mat, mat_dim[0], mat_dim[1]])
    coeffs = [dwt_coeffs[0,:,:], []]
    for i in range(1,num_mat):
        coeffs[1].append(dwt_coeffs[i,:,:])
    coeffs = tuple(coeffs)
    im = pywt.idwt2(coeffs, wavelet_base)
    return im


def callbackFunc(elementsPerPigment, F, a, b, c, d, numPigments, numElements, coeffs):
    ERR_C.append(ERR_C_LAST[0])
    ERR_A.append(ERR_A_LAST[0])
    ERR_K_MEANS.append(ERR_K_MEANS_LAST[0])
    ERR_PI.append(ERR_PI_LAST[0])

    if len(ERR_C)%10 == 1:
        num_it = len(ERR_C)
        dumpResults([elementsPerPigment,ERR_C, ERR_A, ERR_K_MEANS, ERR_PI], ['elementsPerPigment', 'ERR_C', 'ERR_A', 'ERR_K_MEANS','ERR_PI'], num_it)

    print('C Error: '+  str(ERR_C_LAST[0]))
    print('A Error: ' + str(ERR_A_LAST[0]))
    print('K-means Error: '+ str(ERR_K_MEANS_LAST[0]))
    print('PI Error: ' + str(ERR_PI_LAST[0]))

    print('\n')


def dumpResults(resultList, fileNameList, numIt):
    name_suffix  = '_iteration_' + str(numIt) + '_' + WAVELET_TYPE + '.p'
    for res, fname in zip(resultList, fileNameList):
        pickle.dump(res, open(fname+name_suffix, 'wb'))


def minimize(elementsPerPigment, F, a, b, c, d, numPigments, numDynamicPigments, numElements, coeffs, HSCube):
    dynamicRows = elementsPerPigment[:numDynamicPigments].flatten()
    minimized = scipy.optimize.minimize((lambda arr: cost(arr, F, a, b, c, d, numDynamicPigments, numElements, coeffs, HSCube, elementsPerPigment)), dynamicRows,
    options = {'maxiter':100000}, 
    bounds = scipy.optimize.Bounds(np.zeros(numDynamicPigments*numElements), np.ones(numDynamicPigments*numElements), keep_feasible = True),
    constraints = [{'type':'eq', 'fun': (lambda x: np.sum(np.reshape(x, (numDynamicPigments, numElements)),axis=1) - list(np.ones(numDynamicPigments)))}], #should be changed to numDynamicPigments
    callback = (lambda x: callbackFunc(x, F, a, b, c, d, numPigments, numElements, coeffs)))

    optElementsPerPigment = np.reshape(minimized.x, (numDynamicPigments, numElements))
    elementsPerPigment[:numDynamicPigments] = optElementsPerPigment

    A = findA(coeffs, elementsPerPigment)
    F_hat = DWTInvertMany((A.T.dot(elementsPerPigment)).T,NUM_MAT,MAT_DIM,WAVELET_TYPE)
    now = time.localtime()
    
    #time_stamp = '_' +str(now[0])+f'{now[1]:02}'+f'{now[2]:02}'+f'{now[3]:02}'+f'{now[4]:02}'
    name_suffix = '_' + WAVELET_TYPE +'.p'
    pickle.dump(A, open('A_pigmentProportionPerPixel'+name_suffix, 'wb'))
    pickle.dump(F_hat, open('F_hat_Reconstructed'+name_suffix, 'wb'))
    pickle.dump(elementsPerPigment, open('elementsPerPigment'+name_suffix,'wb'))
    pickle.dump(ERR_C, open('ERR_C_LAST'+name_suffix,'wb'))
    pickle.dump(ERR_A, open('ERR_A_LAST'+name_suffix,'wb'))
    pickle.dump(ERR_K_MEANS, open('ERR_K_MEANS_LAST'+name_suffix,'wb'))
    pickle.dump(ERR_PI, open('ERR_PI_LAST'+name_suffix,'wb'))
    print(minimized.nit) 
    print(minimized.message) 
    #{'type':'ineq', 'fun': lambda x: -x+1.}, {'type':'ineq','fun':lambda x: x - 0.0},
    return elementsPerPigment, F_hat


def normalize(data, dataName):
    '''
    This function normalizes each entry in the map by (entry - mean_entry_value) / standard_deviation
    '''
    mm  = np.median(data) 
    normalized = data -  mm
    ss = np.std(normalized)
    #print('the standard deviation of ' + dataName + ' is '+ str(ss), file = text_file)
    #print('the median of '+ dataName + ' is '+ str(mm), file = text_file)
    #print('\n', file = text_file)
    if ss != 0:
        normalized = normalized/ss 
    return normalized, ss, mm


def check(F_hat, original):
    return np.linalg.norm(F_hat - original)




if __name__ == "__main__":
    #file_path = '/Users/ashleykwon/Desktop/R73_Registered/'
    file_path = '/home/ugrad/akwon216/Desktop/R73/'


    barium_map = np.asarray(Image.open(file_path + "R073d01BaL_32reg.tif"))[113:923, 331:1055]
    #calcium_map = np.asarray(Image.open(file_path + "R073d01CaK_32reg.tif"))[113:923, 331:1055]
    cadmium_map = np.asarray(Image.open(file_path + "R073d01CdL_32reg.tif"))[113:923, 331:1055]
    cobalt_map =  np.asarray(Image.open(file_path +  "R073d01CoK_32reg.tif"))[113:923, 331:1055]
    chromium_map = np.asarray(Image.open(file_path +  "R073d01CrK_32reg.tif"))[113:923, 331:1055]
    copper_map =  np.asarray(Image.open(file_path +  "R073d01CuK_32reg.tif"))[113:923, 331:1055]
    iron_map =   np.asarray(Image.open(file_path +  "R073d01FeK_32reg.tif"))[113:923, 331:1055]
    lead_l_map =   np.asarray(Image.open(file_path +  "R073d01PbL_32reg.tif"))[113:923, 331:1055]
    #lead_m_map =  np.asarray(Image.open(file_path +  "R073d01PbM_32reg.tif"))[113:923, 331:1055]
    manganese_map =  np.asarray(Image.open(file_path +  "R073d01MnK_32reg.tif"))[113:923, 331:1055]
    mercury_map =   np.asarray(Image.open(file_path +  "R073d01HgL_32reg.tif"))[113:923, 331:1055]
    #phosphorus_map = np.asarray(Image.open(file_path +  "R073d01PK_32reg.tif"))[113:923, 331:1055]
    potassium_map =  np.asarray(Image.open(file_path +  "R073d01KK_32reg.tif"))[113:923, 331:1055]
    strontium_map =   np.asarray(Image.open(file_path +  "R073d01SeK_32reg.tif"))[113:923, 331:1055]
    sulfur_map = np.asarray(Image.open(file_path +  "R073d01SK_32reg.tif"))[113:923, 331:1055]
    titaium_map = np.asarray(Image.open(file_path +  "R073d01TiK_32reg.tif"))[113:923, 331:1055]
    zinc_map =  np.asarray(Image.open(file_path + "R073d01ZnK_32reg.tif"))[113:923, 331:1055]



    HSCube = np.asarray(Image.open(file_path + 'R-0073-00-000008_reg.tif').crop((331,113,1055,923)))
    pt.imsave('HSCube.png', HSCube)



    HSCube_normalized = np.zeros((HSCube.shape[0],HSCube.shape[1],HSCube.shape[2]))
    for layer in range(HSCube.shape[2]):
        HSCube_normalized[:,:,layer] = normalize(HSCube[:,:,layer], 'cube')[0]
    

    element_maps_round1 = {'barium':barium_map,
    'cadmium':cadmium_map,
    'chromium':chromium_map,
    'cobalt': cobalt_map, 
    'copper':copper_map,
    'iron': iron_map, 
    'lead_l': lead_l_map, 
    'manganese': manganese_map, 
    'mercury':mercury_map,
    'potassium':potassium_map,
    'sulfur': sulfur_map,
    'titaium':titaium_map,
    'zinc': zinc_map}

    
    IM_DIM = np.array([810,724])
    MAT_DIM = (IM_DIM/2).astype(np.int)
    combined_list = []
    original = []
    for map in list(element_maps_round1.keys()): 
        element_maps_round1[map] = normalize(element_maps_round1[map], map)[0]
        coeffs = waveletTransform(element_maps_round1[map])
        combined_list.append(coeffs)
        original.append(element_maps_round1[map].flatten())
    combinedCoeffs = np.array(combined_list) 
    original = np.array(original) 

    #first n rows of the matrix below should be dynamic pigments
    #in the matrix below, the first row is Cobalt Blue [0., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    # elementsPerPigment = np.array([
    # [0, 1, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0.5],
    # [0., 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    # [1., 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 1],
    # [0., 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0., 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    # [0., 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # [0., 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    # ])

    elementsPerPigment = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],    
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0.5],
    [0., 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [1., 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 1],
    [0., 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0., 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0., 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    ])

    elementsPerPigment = (elementsPerPigment.T/np.sum(elementsPerPigment, axis = 1)).T

    numDynamicPigments = 1
    

    a = 10**(-15)
    b = 10**(-7)
    c = 10**(-5)
    d = 10**(-6)

    numPigments = np.shape(elementsPerPigment)[0]
    numElements = np.shape(elementsPerPigment)[1]
    minimize(elementsPerPigment, original, a, b, c, d, numPigments, numDynamicPigments, numElements, combinedCoeffs, HSCube_normalized)
