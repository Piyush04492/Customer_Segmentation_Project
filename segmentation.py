import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import datetime

def main():
    print("Step 1: Loading the dataset...")
    # Path to the dataset
    data_path = 'data/online_retail_II.xlsx'
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    # Load 'Year 2009-2010' sheet (or both if needed, but one sheet is faster and cleaner for demonstration)
    df = pd.read_excel(data_path, sheet_name="Year 2009-2010")
    print(f"Loaded {len(df)} rows.")

    print("\nStep 2: Cleaning the data...")
    # Drop rows without Customer ID
    df = df.dropna(subset=['Customer ID'])
    
    # Cast Customer ID to integer for clean representation
    df['Customer ID'] = df['Customer ID'].astype(int)
    
    # Keep only positive quantities and unit prices (excludes cancellations and bad entries)
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    
    # Calculate TotalSpend (Monetary value) for each transaction
    df['TotalSum'] = df['Quantity'] * df['Price']
    
    # Save cleaned transaction data
    df.to_csv('outputs/cleaned_data.csv', index=False)
    print(f"Cleaned dataset has {len(df)} rows. Saved to outputs/cleaned_data.csv")

    print("\nStep 3: Calculating RFM metrics...")
    # Define a reference date (1 day after the latest purchase date in the dataset)
    reference_date = df['InvoiceDate'].max() + datetime.timedelta(days=1)
    
    # Aggregate data per customer
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days, # Recency
        'Invoice': 'nunique',                                     # Frequency
        'TotalSum': 'sum'                                         # Monetary
    })
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalSum': 'Monetary'
    }, inplace=True)
    
    # Save raw RFM metrics
    rfm.to_csv('outputs/customer_rfm.csv')
    print(f"Calculated RFM metrics for {len(rfm)} unique customers. Saved to outputs/customer_rfm.csv")

    print("\nStep 4: Handling outliers and standardizing features...")
    # Cap outliers at the 99th percentile to improve clustering performance
    rfm_capped = rfm.copy()
    for col in ['Recency', 'Frequency', 'Monetary']:
        q_limit = rfm[col].quantile(0.99)
        rfm_capped[col] = np.clip(rfm_capped[col], 0, q_limit)

    # Standardize features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_capped)

    print("\nStep 5: Finding optimal clusters (Elbow Method)...")
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)
    
    # Plot the Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss, marker='o', linestyle='--', color='#2c3e50')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.grid(True, linestyle=':', alpha=0.6)
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Elbow Method plot to outputs/figures/elbow_method.png")

    print("\nStep 6: Fitting K-Means with optimal K=4...")
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Save clustered customers
    rfm.to_csv('outputs/clustered_customers.csv')
    print("Saved clustered customer data to outputs/clustered_customers.csv")

    print("\nStep 7: Visualizing the clusters...")
    # Calculate average RFM for each cluster to profile them
    cluster_stats = rfm.groupby('Cluster').mean()
    print("\nCluster Profiles (Averages):")
    print(cluster_stats)

    # Generate 3D scatter plot of clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['#1abc9c', '#3498db', '#e74c3c', '#f1c40f']
    
    for cluster_id in range(optimal_k):
        cluster_data = rfm[rfm['Cluster'] == cluster_id]
        ax.scatter(
            cluster_data['Recency'], 
            cluster_data['Frequency'], 
            cluster_data['Monetary'], 
            c=colors[cluster_id], 
            label=f'Cluster {cluster_id}',
            s=40,
            alpha=0.6
        )
        
    ax.set_xlabel('Recency (Days)')
    ax.set_ylabel('Frequency (Purchases)')
    ax.set_zlabel('Monetary Value ($)')
    ax.set_title('Customer Segments Visualization (3D)', fontsize=14, pad=15)
    ax.legend()
    plt.savefig('outputs/figures/customer_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved customer segment scatter plot to outputs/figures/customer_clusters.png")
    
    print("\nClustering process finished successfully!")

if __name__ == '__main__':
    main()
