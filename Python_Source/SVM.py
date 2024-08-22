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
dataset = 3

if dataset ==1:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/All/'
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/SameRandomState/'
    log_file_name = f"logfile_SVM_{current_time}.log"
    figure_name = f"figure_SVM_{current_time}.png"
    target_name_labels = ['Human', 'Horse', 'Dog', 'Cow']
elif dataset ==2:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/AnimalsOnly/'
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/SameRandomState/'
    log_file_name = f"logfile_SVM_{current_time}.log"
    figure_name = f"figure_SVM_{current_time}.png"
    target_name_labels = ['Horse', 'Dog', 'Cow'] 
elif dataset ==3:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/HumanVsAnimal/'
    #datapath = 'C:/Users/Amisha/Documents/Data/SmallTestData/humanvsanimal/' # Test
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/SameRandomState/'
    log_file_name = f"logfile_SVM_{current_time}.log"
    figure_name = f"figure_SVM_{current_time}.png"
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

print("Dataset: Full ")
print(target_name_labels)
print("\n")
print("Params Used:")
print("C = 0.1")
print("kernel = Linear")
print("\n")

# Standardization scales the features so they have the properties of a standard normal distribution with a mean of 0 and a standard deviation of 1. 
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(train_data)

# # # Use the same scaler to transform the test set
# X_test_scaled = scaler.transform(test_data)

model = SVC(C=0.1,kernel='linear',random_state=42) # No need for gamma
model.fit(train_data,train_label)

predictions = model.predict(test_data)

train_accuracy = model.score(train_data, train_label)
test_accuracy = model.score(test_data, test_label)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

cm = confusion_matrix(test_label, predictions)
print(confusion_matrix(test_label, predictions))

print("\n")
print(classification_report(test_label, predictions, target_names=target_name_labels, zero_division=0))

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_name_labels, yticklabels=target_name_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig(figure_path)

log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__



