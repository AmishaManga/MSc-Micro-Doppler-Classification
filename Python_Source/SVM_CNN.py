import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from sklearn.svm import SVC
import datetime
import sys
import seaborn as sns
import os
import random

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

datapath = 'C:/Users/Amisha/Documents/Data/FinalData/All/'
#datapath = 'C:/Users/Amisha/Documents/Data/SmallTestData/humanvsanimals/'
log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/SameRandomState/'
log_file_name = f"SVM_CNN{current_time}.log"
figure_name = f"SVM_CNN{current_time}.png"
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

# Train stage 2 as is (horse:0, dog:1, cow:2)
trainingData2 = train_data2.reshape((-1, height, width, 1))
train_dataset2 = tf.data.Dataset.from_tensor_slices((trainingData2, train_label2)).batch(16)

# validation stage 2 as is (horse:0, dog:1, cow:2)
validationData2 = validation_data2.reshape((-1, height, width, 1))
validation_dataset2 = tf.data.Dataset.from_tensor_slices((validationData2, validation_label2)).batch(16)


# Train data stage 1 (human:0, animal:1)

train_label_adjusted = np.where(train_label > 0, 1, 0) # adjust labels to (0 -human, 1- animal)
test_label_adjusted = np.where(test_label > 0, 1, 0) # adjust labels to (0 -human, 1- animal)

model = SVC(C=0.1,kernel='linear',random_state=42) # No need for gamma
model.fit(train_data,train_label_adjusted)

predictions_stage1 = model.predict(test_data)
test_accuracy = model.score(test_data, test_label_adjusted)
train_accuracy = model.score(train_data, train_label_adjusted)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

cm = confusion_matrix(test_label_adjusted, predictions_stage1)
print(confusion_matrix(test_label_adjusted, predictions_stage1))

print("\n")
print(classification_report(test_label_adjusted, predictions_stage1, target_names=['Human', 'Animal'], zero_division=0))

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
test_data_stage2 = test_data[predictions_stage1 == 1]  # Select only data predicted as animals
original_labels_stage2 = test_label[predictions_stage1 == 1] # True labels of the predicted animals (0,1,2,3)

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