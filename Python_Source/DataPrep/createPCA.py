import numpy as np
from sklearn.decomposition import PCA
import mat73
import h5py
import hdf5storage
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the .mat file containing the spectrogram data mat73.loadmat
mat_contents  = mat73.loadmat('C:/Users/Amisha/Documents/Data/FinalData/All/FinalData.mat')

X = mat_contents['X']  # Spectrogram data
Y = mat_contents['Y']  # classes

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=10)  # for example, keeping 2 components n_components=2698
principal_components = pca.fit_transform(X)

# Step 3: Examine the explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

num__components = np.where(cumulative_variance >= 0.99)[0][0] + 1
print(f"Number of components to retain 99% of the variance: {num__components}")

# Visualize explained variance ratio
plt.figure(1)
plt.bar(range(1, num__components + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Explained Variance Ratio per Principal Component')
plt.show()

plt.figure(2)
plt.plot(cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid(True)
plt.show()

# Not Scaled
# Number of components to retain 99% of the variance: 2698
# Number of components to retain 100% of the variance: 5128
# Animal set: Number of components to retain 99% of the variance: 1206

# Scaled: 
# Number of components to retain 95% of the variance: 1640
# Number of components to retain 99% of the variance: 2737
# Number of components to retain 99% of the variance: 1210


# matfiledata = {} # make a dictionary to store the MAT data in
# matfiledata[u'X'] = principal_components
# matfiledata[u'Y'] = Y
# hdf5storage.write(matfiledata, '.', 'All_Data_Pca_10.mat', matlab_compatible=True)


