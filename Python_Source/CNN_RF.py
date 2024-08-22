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
from sklearn.ensemble import RandomForestClassifier

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

datapath = 'C:/Users/Amisha/Documents/Data/FinalData/All/'
log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/SameRandomState/'
log_file_name = f"CNN_RF{current_time}.log"
figure_name = f"CNN_RF{current_time}.png"
target_name_labels1 = ['Human', 'Animal']

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

print("Full")

print("Dataset 1: But its really human vs animal ")
print(target_name_labels1)

train_label_adjusted = np.where(train_label > 0, 1, 0)
test_label_adjusted = np.where(test_label > 0, 1, 0)

# Reshape into image format
height = 256 
width = 624  

# Set the seed for Python's random module
random.seed(42)

# Set the seed for NumPy
np.random.seed(42)

# Set the seed for TensorFlow
tf.random.set_seed(42)

trainingData = train_data.reshape((-1, height, width, 1))
testData = test_data.reshape((-1, height, width, 1))

train_dataset = tf.data.Dataset.from_tensor_slices((trainingData, train_label_adjusted)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((testData, test_label_adjusted)).batch(16)

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
train_loss1, train_acc1 = model_stage1.evaluate(train_dataset, verbose=3)

# # Generate predictions
predictions1 = model_stage1.predict(test_dataset)
predicted_labels1 = np.argmax(predictions1, axis=1)

# Compute the confusion matrix Test
cm = confusion_matrix(test_label_adjusted, predicted_labels1)
print(cm)

print(classification_report(test_label_adjusted, predicted_labels1, target_names=target_name_labels1, zero_division=0))

print("Test Accuracy CNN Model 1")
print(test_acc1)

print("Train Accuracy CNN Model 1")
print(train_acc1)
print(model_stage1.summary())

print("\n")
print("=====================================================================================================")
print("\n")


datapath2 = 'C:/Users/Amisha/Documents/Data/FinalData/AnimalsOnly/'
target_name_labels2 =  ['Horse - 0', 'Dog - 1', 'Cow - 2'] 

# Prepare Training Data
train_data2 = np.load(datapath2 + 'Train_Data.npy')
train_label2 = np.load(datapath2 +'Train_Label.npy')

train_label2 = train_label2 +1

print("Params Used Model 2:")
print("n_estimators: 100")
print("max_depth: 40")
print("min_samples_split: 2")
print("min_samples_leaf: 1")
print("max_features: sqrt")

model2 = RandomForestClassifier(n_estimators = 100, max_depth = 40, min_samples_split = 2, min_samples_leaf = 1, max_features='sqrt')
model2.fit(train_data2,train_label2)

# Stage 2: Prepare test data based on stage 1 predictions
test_data_stage2 = test_data[predicted_labels1 == 1]  # Predicted as animals by stage 1
original_labels_stage2 = test_label[predicted_labels1 == 1]  # True labels of the predicted animals

predictions2 = model2.predict(test_data_stage2)
test_accuracy2 = model2.score(test_data_stage2, original_labels_stage2)
train_accuracy2 = model2.score(train_data2, train_label2)
print(f"Test Accuracy: {test_accuracy2 * 100:.2f}%")
print(f"Train Accuracy: {train_accuracy2 * 100:.2f}%")

labels = [0 ,1 ,2 ,3]

# Part 2, so now we have classified human/animal

cm2 = confusion_matrix(original_labels_stage2, predictions2, labels = labels)
print(cm2)

print("\n")
print(classification_report(original_labels_stage2, predictions2, labels= labels, target_names=['Human (Misclassified)', 'Horse', 'Dog', 'Cow'], zero_division=0))

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=['Human (Misclassified)', 'Horse', 'Dog', 'Cow'], yticklabels=['Human (Misclassified)', 'Horse', 'Dog', 'Cow'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(figure_path)

log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__