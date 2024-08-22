import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import sys
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

'''
1: All
2: Animals Only
3: Humans vs Animals
'''

datapath = 'C:/Users/Amisha/Documents/Data/FinalData/All/'
log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/SameRandomState/'
log_file_name = f"SVM_TwoStage{current_time}.log"
figure_name = f"SVM_TwoStage{current_time}.png"
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

datapath2 = 'C:/Users/Amisha/Documents/Data/FinalData/AnimalsOnly/'
target_name_labels2 =  ['Horse - 0', 'Dog - 1', 'Cow - 2'] 

train_data2 = np.load(datapath2 + 'Train_Data.npy')
train_label2 = np.load(datapath2 +'Train_Label.npy')

print ("Adding 1 to labels")
print("Dataset2: Horse - 1, Dog - 2 and Cow - 3")

train_label2 = train_label2 +1

model2 = SVC(C=0.1,kernel='linear',random_state=42) # No need for gamma
model2.fit(train_data2,train_label2)

# Stage 2: Prepare test data based on stage 1 predictions
test_data_stage2 = test_data[predictions_stage1 == 1]  # Predicted as animals by stage 1
original_labels_stage2 = test_label[predictions_stage1 == 1]  # True labels of the predicted animals

predictions2 = model2.predict(test_data_stage2)
test_accuracy2 = model2.score(test_data_stage2, original_labels_stage2)
train_accuracy2 = model.score(train_data2, train_label2)
print(f"Test Accuracy: {test_accuracy2 * 100:.2f}%")
print(f"Train Accuracy: {train_accuracy2 * 100:.2f}%")

labels = [0 ,1 ,2 ,3]

# Part 2, so now we have classified human/animal
cm = confusion_matrix(original_labels_stage2, predictions2, labels = labels)
print(cm)

print("\n")
print(classification_report(original_labels_stage2, predictions2, labels= labels, target_names=['Human (Misclassified)', 'Horse', 'Dog', 'Cow'], zero_division=0))

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human (Misclassified)', 'Horse', 'Dog', 'Cow'], yticklabels=['Human (Misclassified)', 'Horse', 'Dog', 'Cow'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(figure_path)

log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__



