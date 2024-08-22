import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import sys
import os
import datetime

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

'''
1: All
2: Animals Only
3: Humans vs Animals
'''
dataset = 3

if dataset ==1:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/All/'
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/All/'
    log_file_name = f"logfile_RF_All_{current_time}.log"
    figure_name = f"figure_RF_All_{current_time}.png"
    target_name_labels = ['Human', 'Horse', 'Dog', 'Cow']
elif dataset ==2:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/AnimalsOnly/'
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/AnimalsOnly/'
    log_file_name = f"logfile_RF_AnimalsOnly_{current_time}.log"
    figure_name = f"figure_RF_AnimalsOnly_{current_time}.png"
    target_name_labels = ['Horse', 'Dog', 'Cow'] 
elif dataset ==3:
    datapath = 'C:/Users/Amisha/Documents/Data/FinalData/HumanVsAnimal/'
    #datapath = 'C:/Users/Amisha/Documents/Data/SmallTestData/humanvsanimal/' # Test
    log_directory = 'C:/Users/Amisha/Documents/MastersGit/Masters/Results/HumanVsAnimal/'
    log_file_name = f"logfile_RF_HumanVsAnimal_{current_time}.log"
    figure_name = f"figure_RF_HumanVsAnimal_{current_time}.png"
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

print("Params Used:")
print("n_estimators: 100")
print("max_depth: 10")
print("min_samples_split: 5")
print("min_samples_leaf: 2")
print("max_features: sqrt")

rf = RandomForestClassifier(n_estimators = 100, max_depth = 10, min_samples_split = 5, min_samples_leaf = 2, max_features='sqrt')
rf.fit(train_data, train_label)

y_pred = rf.predict(test_data)

# Evaluate the model
print("Accuracy:", accuracy_score(test_label, y_pred))

# Calculate and print the confusion matrix
cm = confusion_matrix(test_label, y_pred)
print(cm)

# Calculate accuracies
train_accuracy = rf.score(train_data, train_label)
test_accuracy = rf.score(test_data, test_label)

# Print accuracies
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\n")
print(classification_report(test_label, y_pred, target_names=target_name_labels, zero_division=0))

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



