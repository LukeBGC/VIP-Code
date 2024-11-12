import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Trial 1 input
iMU_1A = pd.read_csv(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 1\A_D422CD00606D_20240314_113325.csv', skiprows = 7)
iMU_1B = pd.read_csv(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 1\B_D422CD006071_20240314_113325.csv', skiprows = 7)
iMU_1C = pd.read_csv(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 1\C_D422CD006078_20240314_113325.csv', skiprows = 7)

#Trial 2 input
iMU_2A = pd.read_csv(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 2\A_D422CD00606D_20240314_114057.csv', skiprows = 7)
iMU_2B = pd.read_csv(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 2\B_D422CD006071_20240314_114057.csv', skiprows = 7)
iMU_2C = pd.read_csv(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 2\C_D422CD006078_20240314_114057.csv', skiprows = 7)

#Trial 3 input
iMU_3A = pd.read_csv(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 3\A_D422CD00606D_20240314_114443.csv', skiprows = 7)
iMU_3B = pd.read_csv(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 3\B_D422CD006071_20240314_114443.csv', skiprows = 7)
iMU_3C = pd.read_csv(r'C:\Users\ldavi\OneDrive\Desktop\Code\VIP Code\IMU Exported Data\Drink Task\Trial 3\C_D422CD006078_20240314_114443.csv', skiprows = 7)

def extract_features(df):
    features = {}
    
    # Statistical features
    for col in ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']:
        features[f'{col}_mean'] = np.mean(df[col])
        features[f'{col}_std'] = np.std(df[col])
        features[f'{col}_var'] = np.var(df[col])
    
    # Jerk (rate of change of acceleration)
    df['jerk_x'] = np.diff(df['Acc_X'], prepend=0)
    df['jerk_y'] = np.diff(df['Acc_Y'], prepend=0)
    df['jerk_z'] = np.diff(df['Acc_Z'], prepend=0)
    features['jerk_x_mean'] = np.mean(df['jerk_x'])
    features['jerk_y_mean'] = np.mean(df['jerk_y'])
    features['jerk_z_mean'] = np.mean(df['jerk_z'])
    
    # Frequency domain features
    for col in ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']:
        fft_vals = fft(df[col])
        features[f'{col}_fft_mean'] = np.mean(np.abs(fft_vals))
        features[f'{col}_fft_var'] = np.var(np.abs(fft_vals))
    
    return features

# Define a function to extract features from all IMUs for a given trial
def combine_imu_features(imu_a, imu_b, imu_c):
    features_a = extract_features(imu_a)
    features_b = extract_features(imu_b)
    features_c = extract_features(imu_c)
    
    # Combine the features of IMUs A, B, and C into one dictionary
    combined_features = {}
    combined_features.update({f'a_{k}': v for k, v in features_a.items()})
    combined_features.update({f'b_{k}': v for k, v in features_b.items()})
    combined_features.update({f'c_{k}': v for k, v in features_c.items()})
    
    return combined_features

# Create a feature matrix for all trials
feature_matrix = []

# Extract features for each trial (trial 1, trial 2, trial 3)
feature_matrix.append(combine_imu_features(iMU_1A, iMU_1B, iMU_1C))
feature_matrix.append(combine_imu_features(iMU_2A, iMU_2B, iMU_2C))
feature_matrix.append(combine_imu_features(iMU_3A, iMU_3B, iMU_3C))

# Convert feature matrix to DataFrame
X = pd.DataFrame(feature_matrix)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering (with 3 clusters for 3 trials)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

# Print the cluster labels
print("Cluster labels for each trial:")
print(cluster_labels)

# Use PCA to visualize the clusters in 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=100)
plt.title('K-Means Clusters of IMU Data Across Trials')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Trial # (-1)')
plt.show()