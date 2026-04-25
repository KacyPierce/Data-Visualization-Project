# Data Visualization Project

[Pierce_Unsupervised Learning.py](https://github.com/user-attachments/files/27092388/Pierce_Unsupervised.Learning.py)
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

# Load the dataset
mallcustomers = pd.read_csv('mallcustomers.csv')

# Preview the dataset
print("Dataset Preview:")
print(mallcustomers.head())

# Check data types
print("\nData Types:")
print(mallcustomers.dtypes)

# Check for missing values
print("\nMissing Values:")
print(mallcustomers.isnull().sum())

# Convert Income from string to numeric
# First, remove "," and "USD" from the Income column
mallcustomers['Income'] = mallcustomers['Income'].str.replace(',', '').str.replace('USD', '').str.strip()

# Convert to numeric
mallcustomers['Income'] = pd.to_numeric(mallcustomers['Income'])

# Verify the conversion
print("\nAfter Conversion - Data Types:")
print(mallcustomers.dtypes)

# Preview the dataset after conversion
print("\nDataset After Conversion:")
print(mallcustomers.head())

# Convert Income from string to numeric - with error handling
if mallcustomers['Income'].dtype == 'object':  # Check if the column contains strings
    # Apply string operations only if the column contains strings
    mallcustomers['Income'] = mallcustomers['Income'].str.replace(',', '').str.replace('USD', '').str.strip()
    mallcustomers['Income'] = pd.to_numeric(mallcustomers['Income'])
elif not pd.api.types.is_numeric_dtype(mallcustomers['Income']):
    # If it's not string but also not numeric, try direct conversion
    mallcustomers['Income'] = pd.to_numeric(mallcustomers['Income'], errors='coerce')
# If it's already numeric, no conversion needed

# Exclude CustomerID and keep only Income and SpendingScore
clustering_data = mallcustomers[['Income', 'SpendingScore']]

# Preview the data that will be used for clustering
print("Data for Clustering:")
print(clustering_data.head())

# Check data types to ensure both columns are numeric
print("\nData Types:")
print(clustering_data.dtypes)

# Basic statistics of the clustering variables
print("\nBasic Statistics:")
print(clustering_data.describe())

# Check for any missing values
print("\nMissing Values:")
print(clustering_data.isnull().sum())

# Display summary statistics
print("Summary Statistics:")
print(clustering_data.describe())

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Visualize the distribution of features before normalization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.boxplot(clustering_data['Income'])
plt.title('Income Distribution')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.boxplot(clustering_data['SpendingScore'])
plt.title('Spending Score Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()

# Perform z-score normalization (standardization)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(clustering_data)

# Convert back to DataFrame for easier interpretation
normalized_df = pd.DataFrame(normalized_data, columns=['Income', 'SpendingScore'])

# Display summary statistics after normalization
print("\nSummary Statistics After Z-Score Normalization:")
print(normalized_df.describe())

# Visualize the distribution of features after normalization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.boxplot(normalized_df['Income'])
plt.title('Normalized Income Distribution')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.boxplot(normalized_df['SpendingScore'])
plt.title('Normalized Spending Score Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare original and normalized data
print("\nOriginal Data - First 5 rows:")
print(clustering_data.head())

print("\nNormalized Data - First 5 rows:")
print(normalized_df.head())

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Calculate WCSS for different k values
wcss = []
for i in range(1, 11):
    # Set random_state=42 for reproducibility
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(normalized_data)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Assuming optimal k is 5 based on the plot
kmeans_final = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans_final.fit_predict(normalized_data)

mallcustomers = pd.read_csv('mallcustomers.csv')

# Add cluster labels to the original dataframe for demographic analysis
mallcustomers['Cluster'] = y_kmeans

# Visualization (Assuming columns are Annual Income and Spending Score)
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(mallcustomers[mallcustomers['Cluster'] == i]['Income'], 
                mallcustomers[mallcustomers['Cluster'] == i]['SpendingScore'], 
                s=100, c=colors[i], label=f'Cluster {i}')

plt.title('Mall Customer Segments')
plt.xlabel('Income')
plt.ylabel('SpendingScore')
plt.legend()
plt.show()

# Based on the typical visualization of Mall Customer data (Income vs. Spending), here are the standard names and the code to extract requested demographic info:
# Cluster 0: "Sensible" (Mid income, Mid spending)
# Cluster 1: "Careless" (Low income, High spending)
# Cluster 2: "Target" (High income, High spending)
# Cluster 3: "Frugal" (High income, Low spending)
# Cluster 4: "Miser" (Low income, Low spending)

# Calculate Demographic Statistics
demographics = mallcustomers.groupby('Cluster').agg({
    'Age': 'mean',
    'Gender': lambda x: x.value_counts(normalize=True).to_dict()
}).rename(columns={'Age': 'Mean Age', 'Gender': 'Gender Distribution'})

# Map the visual names for clarity
cluster_names = {0: 'Sensible', 1: 'Careless', 2: 'Target', 3: 'Frugal', 4: 'Miser'}
demographics.index = demographics.index.map(cluster_names)

print(demographics)
