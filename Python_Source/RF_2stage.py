import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import datetime
import sys
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

'''
1: All
2: Animals Only
3: Humans vs Animals
'''

datapath = 'C:/Users/Amisha/Documents/Data/FinalData/All/'
log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/SameRandomState/'
log_file_name = f"RF_TwoStage{current_time}.log"
figure_name = f"RF_TwoStage{current_time}.png"
target_name_labels1 = ['Human', 'Horse', 'Dog', 'Cow']

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

print("Params Used Model 1:")
print("n_estimators: 300")
print("max_depth: 40")
print("min_samples_split: 10")
print("min_samples_leaf: 1")
print("max_features: log2")

rf = RandomForestClassifier(n_estimators = 300, max_depth = 40, min_samples_split = 10, min_samples_leaf = 1, max_features='log2')
rf.fit(train_data, train_label_adjusted)

predictions_stage1 = rf.predict(test_data)

# Evaluate the model
print("Accuracy:", accuracy_score(test_label_adjusted, predictions_stage1))

test_accuracy = rf.score(test_data, test_label_adjusted)
train_accuracy = rf.score(train_data, train_label_adjusted)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

cm = confusion_matrix(test_label_adjusted, predictions_stage1)
print(cm)

print("\n")
print(classification_report(test_label_adjusted, predictions_stage1, target_names=['Human', 'Animal'], zero_division=0))

datapath2 = 'C:/Users/Amisha/Documents/Data/FinalData/AnimalsOnly/'
target_name_labels2 =  ['Horse - 0', 'Dog - 1', 'Cow - 2'] 

train_data2 = np.load(datapath2 + 'Train_Data.npy')
train_label2 = np.load(datapath2 +'Train_Label.npy')

print ("Adding 1 to labels")
print("Dataset2: Horse - 1, Dog - 2 and Cow - 3")

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
test_data_stage2 = test_data[predictions_stage1 == 1]  # Predicted as animals by stage 1
original_labels_stage2 = test_label[predictions_stage1 == 1]  # True labels of the predicted animals

predictions2 = model2.predict(test_data_stage2)
print("Accuracy:", accuracy_score(original_labels_stage2, predictions2))
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



