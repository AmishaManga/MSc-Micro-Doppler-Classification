from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
import imagesc as imagesc
import seaborn as sns
from matplotlib import cm
import nmmn.plots

ClassificationTestData  = loadmat('E:/MastersData/ReadyToClassifyFinal/AugmentedSets/Hann/Hann_DataLinkDog.mat')
index = 3

# Draw Spectrogram
SpecData = ClassificationTestData['X'][index]
reshapedMatrix = SpecData.reshape(256,624)
transposeMatrix = np.transpose(reshapedMatrix)

plt.figure()
img = plt.imshow(reshapedMatrix,cmap='jet', aspect = 'auto', extent=(0, 7, -4.6, 4.6))
img.set_clim(-30, -10)
plt.colorbar()
plt.title("Spectrogram")
plt.ylabel('Velocity(m/s)')
plt.xlabel('Time (s)')
plt.show()