import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
import datetime
import sys
import seaborn as sns
import os
import random
from tensorflow.keras.utils import to_categorical

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

datapath = 'C:/Users/Amisha/Documents/Data/FinalData/All/'
#datapath = 'C:/Users/Amisha/Documents/Data/SmallTestData/humanvsanimals/'
log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/SameRandomState/'
log_file_name = f"CNN_TwoStage{current_time}.log"
figure_name = f"CNN_TwoStage{current_time}.png"
target_name_labels1 = ['Human - 0','Animal - 1']

train_data = np.load(datapath + 'Train_Data.npy')
test_data = np.load(datapath +'Test_Data.npy')
train_label = np.load(datapath +'Train_Label.npy')
test_label = np.load(datapath +'Test_Label.npy')

datapath2 = 'C:/Users/Amisha/Documents/Data/FinalData/AnimalsOnly/'
target_name_labels2 =  ['Horse - 0', 'Dog - 1', 'Cow - 2'] 

train_data2 = np.load(datapath2 + 'Train_Data.npy')
train_label2 = np.load(datapath2 +'Train_Label.npy')

# Prepare Validation Test Data for the model: Use test data from animal set to validate
validation_data2 = np.load(datapath2 + 'Test_Data.npy')
validation_label2 = np.load(datapath2 + 'Test_Label.npy')

# Open a file to redirect the output
log_file_path = os.path.join(log_directory, log_file_name)
figure_path = os.path.join(log_directory, figure_name)
log_file = open(log_file_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

# Reshape into image format
height = 256 
width = 624  

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Train data stage 1 (human:0, animal:1)
trainingData = train_data.reshape((-1, height, width, 1))
train_label_adjusted = np.where(train_label > 0, 1, 0) # adjust labels to (0 -human, 1- animal)
train_dataset = tf.data.Dataset.from_tensor_slices((trainingData, train_label_adjusted)).batch(16)

# Test Data Stage 1 (human:0, animal:1)
testData = test_data.reshape((-1, height, width, 1))
test_label_adjusted = np.where(test_label > 0, 1, 0) # adjust labels to (0 -human, 1- animal)
test_dataset = tf.data.Dataset.from_tensor_slices((testData, test_label_adjusted)).batch(16)


# Train stage 2 as is (horse:0, dog:1, cow:2)
trainingData2 = train_data2.reshape((-1, height, width, 1))
train_dataset2 = tf.data.Dataset.from_tensor_slices((trainingData2, train_label2)).batch(16)

# validation stage 2 as is (horse:0, dog:1, cow:2)
validationData2 = validation_data2.reshape((-1, height, width, 1))
validation_dataset2 = tf.data.Dataset.from_tensor_slices((validationData2, validation_label2)).batch(16)

# Building the second model for subclassification
model_stage1 = models.Sequential()
model_stage1.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_stage1.add(layers.MaxPooling2D((2, 2)))
model_stage1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_stage1.add(layers.MaxPooling2D((2, 2)))
model_stage1.add(layers.Flatten())
model_stage1.add(layers.Dense(128, activation='relu'))
model_stage1.add(layers.Dropout(0.5))
model_stage1.add(layers.Dense(len(target_name_labels1), activation='softmax'))

model_stage1.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

history_1 = model_stage1.fit(train_dataset, epochs=10 , validation_data=test_dataset)

test_loss1, test_acc1 = model_stage1.evaluate(test_dataset, verbose=3)

# # Generate predictions
predictions1 = model_stage1.predict(test_dataset)
predicted_labels1 = np.argmax(predictions1, axis=1)

# Compute the confusion matrix Test
cm = confusion_matrix(test_label_adjusted, predicted_labels1)
print(cm)

print(classification_report(test_label_adjusted, predicted_labels1, target_names=target_name_labels1, zero_division=0))

print("Test Accuracy CNN Model 1")
print(test_acc1)

print("\n")
print("=====================================================================================================")
print("\n")


# Building the second model for subclassification
model_stage2 = models.Sequential()
model_stage2.add(layers.Conv2D(32, (3, 3), activation='relu'))
model_stage2.add(layers.MaxPooling2D((2, 2)))
model_stage2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_stage2.add(layers.MaxPooling2D((2, 2)))
model_stage2.add(layers.Flatten())
model_stage2.add(layers.Dense(128, activation='relu'))
model_stage2.add(layers.Dropout(0.5))
model_stage2.add(layers.Dense(len(target_name_labels2), activation='softmax'))

model_stage2.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

history_2 = model_stage2.fit(train_dataset2, epochs=10 , validation_data=validation_dataset2)

# Prepare test data based on stage 1 predictions
test_data_stage2 = test_data[predicted_labels1 == 1]  # Select only data predicted as animals
original_labels_stage2 = test_label[predicted_labels1 == 1] # True labels of the predicted animals (0,1,2,3)

testData2 = test_data_stage2.reshape((-1, height, width, 1))

final_test_dataset = tf.data.Dataset.from_tensor_slices((testData2)).batch(16) # for all species, original labels

final_test_dataset_labels = tf.data.Dataset.from_tensor_slices((testData2, original_labels_stage2)).batch(16) # for all species, original labels

# # Generate predictions and handle label mappings for clarity
predictions2 = model_stage2.predict(testData2) # (0,1,2)
predicted_labels2 = np.argmax(predictions2, axis=1)
predicted_labels_original = predicted_labels2 + 1 # (1,2,3)

# Assuming original_labels_stage2 are your true labels already loaded correctly
correct_predictions = (predicted_labels_original == original_labels_stage2)
accuracy = np.mean(correct_predictions)
print(f"Calculated Accuracy: {accuracy * 100:.2f}%")

labels = [0, 1 ,2 ,3 ] 

# Compute the confusion matrix Test
cm2 = confusion_matrix(original_labels_stage2, predicted_labels_original, labels = labels)
print(cm2)

print(classification_report(original_labels_stage2, predicted_labels_original, labels= labels, target_names=['Human (Misclassified)', 'Horse', 'Dog', 'Cow'], zero_division=0))

# # Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=['Human (Misclassified)', 'Horse', 'Dog', 'Cow'], yticklabels=['Human (Misclassified)', 'Horse', 'Dog', 'Cow'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(figure_path)

print("Model 2 Summary")
print(model_stage2.summary())

log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__