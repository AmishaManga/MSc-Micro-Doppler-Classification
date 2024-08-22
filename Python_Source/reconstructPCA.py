import numpy as np
from sklearn.decomposition import PCA
import mat73
import h5py
import hdf5storage
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

datapath = 'C:/Users/AmishaManga/Documents/Data/FinalData.mat'

# Load the .mat file containing the spectrogram data mat73.loadmat
mat_contents  = mat73.loadmat(datapath)

X = mat_contents['X']  # Spectrogram data
Y = mat_contents['Y']  # classes

#pca = PCA(n_components=2698)  # for example, keeping 2 components n_components=2698
#reduced_data = pca.fit_transform(X)

# # Reconstruct the data from reduced dimensions
#reconstructed_data = pca.inverse_transform(reduced_data)

# # # Reshape the reconstructed data to the original shape of the spectrogram
#reconstructed_spectrogram = reconstructed_data  # Transpose back to the original shape

# # Visualize the reconstructed spectrogram (plotting a random spectrogram)
index = 667  # Change this to visualize a different spectrogram
#SpecData = reconstructed_spectrogram[index]
SpecDataOriginal = X[index]

#reshapedMatrix = SpecData.reshape(256,624)
originalMatrix = SpecDataOriginal.reshape(256,624)

plt.figure(figsize=(8, 6))
img = plt.imshow(originalMatrix,cmap='jet', aspect = 'auto', extent=(0, 7, -4.6, 4.6))
img.set_clim(-30, -10)
cbar = plt.colorbar()
cbar.set_ticks(np.arange(-30, -9, 2))  # Set ticks with integer values
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))  # Format ticks without decimal points
plt.title("Original Number Components = 159744")
plt.ylabel('Velocity (m/s)')
plt.xlabel('Time (s)')

# Set y-tick and x-tick labels
yticklabels = np.linspace(-4.6, 4.6, 5)
xticklabels = np.linspace(0, 7, 8)

plt.yticks(yticklabels)
plt.xticks(xticklabels)
plt.savefig('numComponents_159744.pdf')
plt.savefig('numComponents_159744.svg')

plt.show()