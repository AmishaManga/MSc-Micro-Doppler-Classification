from scipy.io import loadmat
import pandas as pd
import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from array import array
import seaborn as sns
import nmmn.plots 
import mat73

# Load the .mat file containing the spectrogram data mat73.loadmat
mat_contents  = mat73.loadmat('E:/MastersData/ReadyToClassifyFinal/FinalData.mat')

X = mat_contents['X']  # Spectrogram data

# Assuming X is your dataset
# Step 1: Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Perform PCA
pca = PCA()
pca.fit(X)

# Step 3: Apply the Kaiser Criterion
eigenvalues = pca.explained_variance_
components_to_retain = sum(eigenvalues > 1)

print(f"Number of components to retain according to the Kaiser Criterion: {components_to_retain}")


# # Perform PCA
# #num_components = 2698  # number of components to retain 99% of the variance: 2703
# pca = PCA()

# # Get the transformed data (reduced dimensions)
# reduced_data = pca.fit_transform(X)

# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# # Assuming cumulative_variance is already calculated as np.cumsum(pca.explained_variance_ratio_)
# print("Maximum cumulative variance:", cumulative_variance.max())

# num_components = np.where(cumulative_variance >= 0.99)[0][0] + 1
# print(f"Number of components to retain 99% of the variance: {num_components}")

# # # After fitting PCA, plot explained variance ratio
# plt.figure(figsize=(8, 6))
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(cumulative_variance)
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Variance Explained by Components')

# # Highlight the point where 99% variance is retained
# target_variance = 0.99
# # Check if 99% variance is achieved
# indices_where_99_is_reached = np.where(cumulative_variance >= target_variance)[0]
# if indices_where_99_is_reached.size > 0:
#     num_components_99_var = indices_where_99_is_reached[0] + 1  # Correct index to reflect counting from 1
#     plt.axvline(x=num_components_99_var, color='r', linestyle='--')
#     plt.axhline(y=target_variance, color='r', linestyle='--')
#     plt.text(num_components_99_var, target_variance, f'  {num_components_99_var} Components 99% Variance', verticalalignment='bottom', horizontalalignment='right')
# else:
#     print("99% variance is not achieved within the given components.")

# plt.savefig('explained_variance_.png')


# # Use reduced_data for further analysis or classification tasks

# # Perform PCA and obtain 'reduced_data' (as done in the previous steps)

# # Reconstruct the data from reduced dimensions
# reconstructed_data = pca.inverse_transform(reduced_data)

# # # Reshape the reconstructed data to the original shape of the spectrogram
# reconstructed_spectrogram = reconstructed_data  # Transpose back to the original shape

# # Visualize the reconstructed spectrogram (plotting a random spectrogram)
# index = 458  # Change this to visualize a different spectrogram
# SpecData = reconstructed_spectrogram[index]
# SpecDataOriginal = X[16]

# reshapedMatrix = SpecData.reshape(256,624)
# originalMatrix = SpecDataOriginal.reshape(256,624)

# plt.figure(figsize=(8, 6))
# img = plt.imshow(reshapedMatrix,cmap='jet', aspect = 'auto', extent=(0, 7, -4.6, 4.6))
# img.set_clim(-30, -10)
# plt.colorbar()
# plt.title("Spectrogram")
# plt.ylabel('Velocity (m/s)')
# plt.xlabel('Time (s)')
# plt.savefig('spectrogram_pca.png')

# plt.figure(figsize=(8, 6))
# img = plt.imshow(originalMatrix,cmap='jet', aspect = 'auto', extent=(0, 7, -4.6, 4.6))
# img.set_clim(-30, -10)
# plt.colorbar()
# plt.title("Spectrogram")
# plt.ylabel('Velocity (m/s)')
# plt.xlabel('Time (s)')
# plt.savefig('spectrogram_original.png')

