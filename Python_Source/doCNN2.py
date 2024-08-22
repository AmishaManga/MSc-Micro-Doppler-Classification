import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import sys
import os
import datetime
import random


current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

'''
1: All
2: Animals Only
3: Humans vs Animals
'''
dataset = 1

if dataset ==1:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/All/'
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/SameRandomState/'
    log_file_name = f"logfile_CNN2_All_{current_time}.log"
    figure_name = f"figure_CNN2_All_{current_time}.png"
    target_name_labels = ['Human', 'Horse', 'Dog', 'Cow']
elif dataset ==2:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/AnimalsOnly/'
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/SameRandomState/'
    log_file_name = f"logfile_CNN2_AnimalsOnly_{current_time}.log"
    figure_name = f"figure_CNN2_AnimalsOnly_{current_time}.png"
    target_name_labels = ['Horse', 'Dog', 'Cow'] 
elif dataset ==3:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/HumanVsAnimal/'
    #datapath = 'C:/Users/Amisha/Documents/Data/SmallTestData/humanvsanimal/' # Test
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/SameRandomState/'
    log_file_name = f"logfile_CNN2_HumanVsAnimal_{current_time}.log"
    figure_name = f"figure_CNN2_HumanVsAnimal_{current_time}.png"
    target_name_labels = ['Human', 'Animal']
else:
    print("Error: Choose a dataset")

train_data = np.load(datapath + 'Train_Data.npy')
test_data = np.load(datapath +'Test_Data.npy')
train_label = np.load(datapath +'Train_Label.npy')
test_label = np.load(datapath +'Test_Label.npy')

# Open a file to redirect the output
log_file_path = os.path.join(log_directory, log_file_name)
figure_path = os.path.join(log_directory, figure_name)
log_file = open(log_file_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

print("Dataset: ")
print(target_name_labels)

# Initialize the TensorBoard callback
#tensorboard_callback = TensorBoard(log_dir=log_directory, histogram_freq=1)

# Reshape into image format
height = 256 
width = 624  

print(train_data.shape)

# Set the seed for Python's random module
random.seed(42)

# Set the seed for NumPy
np.random.seed(42)

# Set the seed for TensorFlow
tf.random.set_seed(42)


trainingData = train_data.reshape((-1, height, width, 1))
testData = test_data.reshape((-1, height, width, 1))

train_dataset = tf.data.Dataset.from_tensor_slices((trainingData, train_label)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((testData, test_label)).batch(16)

# Building the second model for subclassification
model_stage2 = models.Sequential()
model_stage2.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_stage2.add(layers.MaxPooling2D((2, 2)))
model_stage2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_stage2.add(layers.MaxPooling2D((2, 2)))
model_stage2.add(layers.Flatten())
model_stage2.add(layers.Dense(128, activation='relu'))
model_stage2.add(layers.Dropout(0.5))
model_stage2.add(layers.Dense(len(target_name_labels), activation='softmax'))

model_stage2.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

history_2 = model_stage2.fit(train_dataset, epochs=10 , validation_data=test_dataset)

test_loss2, test_acc2 = model_stage2.evaluate(test_dataset, verbose=3)
train_loss2, train_acc2 = model_stage2.evaluate(train_dataset, verbose=3)

#model_stage2.save('CNN.h5')

# # Generate predictions
predictions2 = model_stage2.predict(test_dataset)
predicted_labels2 = np.argmax(predictions2, axis=1)

# Compute the confusion matrix Test
cm = confusion_matrix(test_label, predicted_labels2)

print(classification_report(test_label, predicted_labels2, target_names=target_name_labels, zero_division=0))


# # Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_name_labels, yticklabels=target_name_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(figure_path)

print("Test Accuracy Model 2")
print(test_acc2)

print("Train Accuracy Model 2")
print(train_acc2)
print(model_stage2.summary())

log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__