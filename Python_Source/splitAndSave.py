from sklearn.model_selection import train_test_split
import numpy as np
from scipy.io import loadmat
import mat73

path = 'C:/Users/Amisha/Documents/Data/SmallTestData/animals/'

# Load .mat file

#data = mat73.loadmat(path + 'All_Data_Pca_10.mat')
data = loadmat(path+ 'FinalData.mat')
#data = loadmat('C:/Users/AmishaManga/Documents/Data/AnimalsvsHuman/PCA/FinalData.mat')
#data = mat73.loadmat('C:/Users/Amisha/Documents/Data/FinalData/All/PCA/Unscaled/All_Data_Pca_Unscaled_2698.mat')

# Spectrogram data
specgramData = data['X']
classes = data['Y'].T

classes = classes.flatten()

# Subtract one so we dealing with 0 and 1 and not 1 and 2
#classes = classes - 1

# Split
train_data, test_data, train_labels, test_labels = train_test_split(specgramData, classes, test_size=0.2)

# Save to npy
np.save(path + 'Train_Data.npy', train_data)
np.save(path + 'Train_Label.npy', train_labels)
np.save(path + 'Test_Data.npy', test_data)
np.save(path + 'Test_Label.npy', test_labels)




