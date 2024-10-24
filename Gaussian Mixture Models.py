import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Function to load and preprocess data
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path, skiprows=7)  # Skip the first 7 rows
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Load and preprocess data for each IMU and trial
imu_A_trial_1 = load_and_preprocess(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 1\A_D422CD00606D_20240314_113325.csv')
imu_A_trial_2 = load_and_preprocess(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 2\A_D422CD00606D_20240314_114057.csv')
imu_A_trial_3 = load_and_preprocess(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 3\A_D422CD00606D_20240314_114443.csv')

imu_B_trial_1 = load_and_preprocess(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 1\B_D422CD006071_20240314_113325.csv')
imu_B_trial_2 = load_and_preprocess(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 2\B_D422CD006071_20240314_114057.csv')
imu_B_trial_3 = load_and_preprocess(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 3\B_D422CD006071_20240314_114443.csv')

imu_C_trial_1 = load_and_preprocess(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 1\C_D422CD006078_20240314_113325.csv')
imu_C_trial_2 = load_and_preprocess(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 2\C_D422CD006078_20240314_114057.csv')
imu_C_trial_3 = load_and_preprocess(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 3\C_D422CD006078_20240314_114443.csv')

# Function to apply GMM and get cluster means
def gmm_and_get_means(data, n_components=2):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data)
    labels = gmm.predict(data)
    cluster_means = pd.DataFrame(data).groupby(labels).mean()
    return cluster_means

# Analyze each IMU across trials
results = {}
imu_names = ['A', 'B', 'C']
trials = [imu_A_trial_1, imu_A_trial_2, imu_A_trial_3, 
          imu_B_trial_1, imu_B_trial_2, imu_B_trial_3, 
          imu_C_trial_1, imu_C_trial_2, imu_C_trial_3]

# Loop through each IMU and trial to store cluster means
for i, imu in enumerate(imu_names):
    results[imu] = {}
    for trial_num in range(1, 4):
        data = trials[(i * 3) + (trial_num - 1)]
        results[imu][f'Trial {trial_num}'] = gmm_and_get_means(data)

# Calculate improvement (Euclidean distance between cluster means across trials)
improvement = {}
for imu in imu_names:
    improvement[imu] = []
    for trial_num in range(1, 3):  # Compare Trial 1 to Trial 2 and Trial 2 to Trial 3
        trial_1_means = results[imu][f'Trial {trial_num}']
        trial_2_means = results[imu][f'Trial {trial_num + 1}']
        # Compute Euclidean distance between means
        distance = np.linalg.norm(trial_2_means.values - trial_1_means.values)
        improvement[imu].append(distance)

# Plot the improvement across trials for each IMU
trials = [1, 2]  # Trial 1 to Trial 2, and Trial 2 to Trial 3

plt.figure(figsize=(10, 6))

# Plot improvement for each IMU
colors = {'A': 'blue', 'B': 'green', 'C': 'red'}
for imu in imu_names:
    plt.plot(trials, improvement[imu], label=f'IMU {imu}', marker='o', color=colors[imu])

# Add labels and title
plt.xlabel('Trial Comparison')
plt.ylabel('Improvement (Euclidean Distance)')
plt.title('Improvement in Movement Quality Across Trials (by IMU)')
plt.xticks(trials, ['Trial 1 to Trial 2', 'Trial 2 to Trial 3'])
plt.legend()
plt.grid(True)

# Show plot
plt.show()