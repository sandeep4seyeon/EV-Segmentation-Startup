import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Extracting dataset - 1
ds1 = pd.read_csv('RS_Session.csv')
print(ds1)

# Extracting dataset - 2
ds2 = pd.read_csv('indian-ev-data.csv')
print(ds2)

# checking the shape (# of rows and columns) of the datasets
print('DS1 Shape: ', ds1.shape)
print('DS2 Shape: ', ds2.shape)

# checking the info (columns, datatypes, nulls) of the datasets
print(' <<< DATASET 1 >>> ')
print(ds1.info())
print('\n <<< DATASET 2 >>>')
print(ds2.info())

# getting a statistical summary of the datasets
s1 = ds1.describe()
s2 = ds2.describe()
print('<<< DATASET 1 >>>', s1, '<<< DATASET 2 >>>', s2)

# Extracting dataset - 3
ds3 = pd.read_excel('Ev Sales.xlsx')
print(ds3)

# 2 wheelers data visualization from dataset 1
# Create a barplot using seaborn
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")  # Optional: Set the style of the plot
# Plot the total number of two-wheeler vehicles using sns.barplot
sns.barplot(x="Two Wheeler", y="State Name", data=ds1, orient="h")

# Customize plot labels and appearance
plt.xlabel("Total two wheeler")
plt.ylabel("State")
plt.title("Total Number of Two-Wheeler ")

# Show the plot
plt.tight_layout()
plt.show()

# 3 wheelers data visualization from dataset 1
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")  # Optional: Set the style of the plot

# Plot the total number of two-wheeler vehicles using sns.barplot
sns.barplot(x="Three Wheeler", y="State Name", data=ds1, orient="h")

# Customize plot labels and appearance
plt.xlabel("Total Number of Three Wheeler Vehicles")
plt.ylabel("State")
plt.title("Number of Three Wheeler by State")

# Show the plot
plt.tight_layout()
plt.show()

# 4 wheelers data visualization from dataset 1
plt.figure(figsize=(8, 6))
sns.barplot(
    data=ds1,
    y=ds1['State Name'].sort_values(ascending=True),  # Ensure that the column name is correct
    x='Four Wheeler',
    hue=ds1['State Name'],  # Assign hue to avoid the FutureWarning
    palette='viridis',
    dodge=False  # Use dodge=False to keep bars unified
)

# Turn off the legend (since each state name is unique, a legend is unnecessary)
plt.legend([], [], frameon=False)

# Rotate y-axis labels for better readability
plt.yticks(rotation=0, ha='right')

# Customize plot labels
plt.xlabel("Number of Four-Wheeler Vehicles")
plt.ylabel("State Name")
plt.title("Number of Four-Wheeler Vehicles by State")
plt.tight_layout()
plt.show()

# Goods and Services Vehicles data visualization from dataset 1
ds1['Goods and Services Vehicles'] = ds1[['Goods Vehicles', 'Public Service Vehicle', 'Other']].sum(axis=1)
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")  # Optional: Set the style of the plot
sns.barplot(x="Goods and Services Vehicles", y="State Name", data=ds1, orient="h")

# Customize plot labels and appearance
plt.xlabel("Total Number of Goods and Services Vehicles")
plt.ylabel("State")
plt.title("Number of Goods and Services Vehicles by State")
plt.tight_layout()
plt.show()

# Sum the total sales for each segment from Apr2017-May23
total_sales = {
    '2 W': ds3['2 W'].sum(),  # Replace 'Segment1' with actual column name
    '3 W': ds3['3 W'].sum(),  # Replace 'Segment2' with actual column name
    '4 W': ds3['4 W'].sum(),  # Replace 'Segment3' with actual column name
    'BUS': ds3['BUS'].sum()   # Replace 'Segment4' with actual column name
}

# Step 3: Convert to percentages
total_sales_series = pd.Series(total_sales)
sales_percentage = total_sales_series / total_sales_series.sum() * 100

# Step 4: Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sales_percentage, labels=sales_percentage.index, autopct='%1.1f%%', startangle=140)
plt.title('Total Sales Percentage by Segment (Apr17-May23)')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

# brand-wise count of EV models
plt.figure(figsize=(12, 8))

# Create the count plot, avoiding the deprecation warning by assigning 'Manufacturer' to hue and setting legend=False
sns.catplot(
    data=ds2,
    x='Manufacturer',
    kind='count',
    hue='Manufacturer',  # Assign hue to avoid the palette warning
    palette='viridis',
    legend=False,  # Turn off the unnecessary legend
    height=6,
    aspect=2
)

# Customize the labels and title
plt.xlabel('Manufacturer')
plt.ylabel('Number of EVs')
plt.title('Number of Electric Vehicles by Manufacturer')

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45, ha='right')

# Adjust the layout to prevent label clipping
plt.tight_layout()

# Show the plot
plt.show()

# Define price ranges and create bins
bins = [60000, 100000, 150000, 200000, 250000, 300000]  # Define your bins
labels = ['60k-100k', '100k-150k', '150k-200k', '200k-250k', '250k-300k']  # Define labels

ds2['Price Range'] = pd.cut(ds2['Price'], bins=bins, labels=labels, right=False)  # Adjust 'Price' to your column name

# Count the number of entries in each price range
price_range_counts = ds2['Price Range'].value_counts().sort_index()

# Create a bar chart for Price Range
plt.figure(figsize=(10, 6))
price_range_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Entries in Each Price Range')
plt.xlabel('Price Range')
plt.ylabel('Number of Entries')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y')
plt.show()

# plotting the price from dataset 2
plt.plot(ds2['Price'], color='c')
plt.xlabel('Number of Samples', family='serif', size=12)
plt.ylabel('Price', family='serif', size=12)
plt.title('Price Comparison', family='serif', size=15, pad=12)
plt.show()

# plotting the price from dataset 2
plt.figure(figsize=(8, 6))
sns.boxplot(data=ds2, y='Price', color='c')
plt.ylabel('Price (Rupees)', family='serif', size=12)
plt.title('Price Distribution', family='serif', size=15, pad=12)
plt.show()

# Analysis of Price for Different Battery Capacity
plt.scatter(ds2['Battery Capacity (kWh)'], ds2['Price'], color='g', alpha=0.7)
plt.xlabel('Battery Capacity (kWh)', family='serif', size=12)
plt.ylabel('Price (Rupees)', family='serif', size=12)
plt.title('Price vs. Battery Capacity', family='serif', size=15, pad=12)
plt.grid(True)
plt.show()

# Charging time visualization from dataset 2
plt.figure(figsize=(6, 8))
sns.barplot(data=ds2, y='Manufacturer', x='Charging Time', errorbar=None, hue='Manufacturer', palette='viridis')
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.xlabel('Charging Time', family='serif', size=12)
plt.ylabel('Manufacturer', family='serif', size=12)
plt.title(label='Charging Time of EVs in India', family='serif', size=15, pad=12)
plt.show()

# Compute the correlation matrix
numeric_data = ds2.select_dtypes(include=['number'])  # This selects only numeric columns
correlation_matrix = numeric_data.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data=correlation_matrix, annot=True, cmap='Purples', cbar=False, square=True, fmt='.2f', linewidths=.3)
plt.title('Correlation Heatmap')
plt.show()

# Define the ranges (bins)
bins = [0, 60, 90, 120, 150, 200]
labels = ['0-60', '60-90', '90-120', '120-150', '150-200']

ds2['Range in kms'] = pd.cut(ds2['Range per Charge (km)'], bins=bins, labels=labels, right=False)

# Group by Manufacturer and Range
brand_range_counts = ds2.groupby(['Manufacturer', 'Range per Charge (km)']).size().unstack(fill_value=0)

# Create a bar plot for brand-wise analysis
plt.figure(figsize=(12, 6))
brand_range_counts.plot(kind='bar', stacked=True, colormap='viridis')

plt.title('Brand-wise Analysis of Value Ranges')
plt.xlabel('Brand')
plt.ylabel('Number of Entries')
plt.xticks(rotation=45)
plt.legend(title='Value Range')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

ds2 = ds2.drop(index=49)

# Select features for clustering
features = ['Top Speed (km/h)', 'Power (HP or kW)', 'Price', 'Charging Time',
            'Range per Charge (km)', 'Battery Capacity (kWh)']  # Replace with your actual feature names

# Handle categorical data
X = ds2[features].copy()

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Determine the optimal number of clusters (Elbow Method)
inertia = []
silhouette_scores = []
k_range = range(2, 11)  # Adjust this range as needed

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot the silhouette scores
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Score Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Step 3: Fit the K-Means Model
optimal_k = 4  # Replace this with the optimal number of clusters you found
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
ds2['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 4: Analyze the Results
# Visualizing the clusters (only using the first two features for 2D plot)
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=ds2['Cluster'], cmap='viridis', marker='o')
plt.title('K-Means Clustering Results')
plt.xlabel('First Feature (scaled)')
plt.ylabel('Second Feature (scaled)')
plt.colorbar(label='Cluster')
plt.show()

# selecting features for building a model
X = ds2[['Year of Manufacture', 'Top Speed (km/h)', 'Power (HP or kW)', 'Price', 'Charging Time',
         'Range per Charge (km)', 'Battery Capacity (kWh)']]

# Step 2: Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check the number of features
n_features = X.shape[1]
print(f"Number of features: {n_features}")

# Step 3: Apply PCA
n_components = min(7, n_features)  # Set number of components to the minimum of 7 or number of features
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=X_pca, columns=[f'Principal Component {i+1}' for i in range(n_components)])

# Step 4: Visualize the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_components + 1), pca.explained_variance_ratio_, marker='o', linestyle='--')
plt.title('Explained Variance Ratio by Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, n_components + 1))
plt.grid(True)
plt.show()

# Step 5: Print the explained variance ratio for only the first 3 components
print("Explained Variance Ratio by First 3 Principal Components:")
print(pca.explained_variance_ratio_[:3])

# Step 6: Fit the model and assign cluster labels based on PCA data
optimal_k = 4  # Set your optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
pca_df['Cluster'] = kmeans.fit_predict(X_pca)  # Use X_pca for clustering

# Step 7: Print the PCA components with correct columns
feature_names = X.columns  # This now has the correct number of columns
pca_components = pd.DataFrame(pca.components_, columns=feature_names,
                              index=[f'Principal Component {i+1}' for i in range(n_components)])
print("\nPCA Components (Weights for Each Feature):")
print(pca_components)

cluster_profiles = {}

# Iterate through each cluster label
for cluster_label in ds2['Cluster'].unique():
    # Filter the data for the current cluster
    cluster_data = ds2[ds2['Cluster'] == cluster_label]
# Step 8: Optional: Visualize the first two PCA components with Clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'],
                      c=pca_df['Cluster'], cmap='viridis', marker='o')
plt.title('PCA of Dataset with K-Means Clustering (First 2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()
# Plot the elbow graph
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)
print("Cluster Centers (scaled):")
print(kmeans.cluster_centers_)

print("\nCluster Distribution:")
print(ds2['Cluster'].value_counts())
# Step 2: Add cluster labels to the dataset
ds2['Cluster'] = kmeans.labels_
plt.figure(figsize=(7, 5))

# Scatter plot of PCA results, color-coded by cluster labels
sns.scatterplot(data=pca_df, x='Principal Component 1', y='Principal Component 2', s=70,
                hue=pca_df['Cluster'], palette='viridis', zorder=2, alpha=.9)

# Plot the K-Means centroids
plt.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
            marker="*", c="r", s=80, label="Centroids")

# Customize the plot's appearance
plt.xlabel('Principal Component 1 (PC1)', family='serif', size=12)
plt.ylabel('Principal Component 2 (PC2)', family='serif', size=12)
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.grid()
plt.tick_params(grid_color='lightgray', grid_linestyle='--', zorder=1)

# Add legend
plt.legend(title='Labels', fancybox=True, shadow=True)

# Set title
plt.title('K-Means Clustering Results', family='serif', size=15)

# Display the plot
plt.show()

cluster_summary = ds2.groupby('Cluster').agg({
    'Battery Capacity (kWh)': 'mean',        # You can use 'median' instead of 'mean' if preferred
    'Range per Charge (km)': 'mean',
    'Charging Time': 'mean',
    'Price': 'mean',
    'Power (HP or kW)': 'mean',
    'Top Speed (km/h)': 'mean'
})
print(cluster_summary)
print("\n")

for cluster_label, profile in cluster_summary.items():
    print(f"Cluster {cluster_label} Profile:")
    for key, value in profile.items():
        print(f"{key}: {value}")
    print("\n")
