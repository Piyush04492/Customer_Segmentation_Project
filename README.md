# Customer Segmentation using RFM and K-Means Clustering

This project implements a simple, end-to-end customer segmentation pipeline on the **Online Retail II** dataset. 
We perform RFM (Recency, Frequency, Monetary) analysis to quantify customer behavior and apply K-Means clustering using `scikit-learn` to group customers into distinct segments.

## Project Structure

```
segmentation/
│
├── data/
│   └── online_retail_II.xlsx         # Raw dataset downloaded from UCI
│
├── notebooks/
│   └── Customer_Segmentation.ipynb   # Interactive analysis and visualization
│
├── outputs/
│   ├── cleaned_data.csv              # Transactions after filtering and cleaning
│   ├── customer_rfm.csv              # Calculated RFM metrics per customer
│   ├── clustered_customers.csv       # Customers with their assigned cluster labels
│   └── figures/
│       ├── elbow_method.png          # Elbow curve for choosing K
│       └── customer_clusters.png     # 3D visualization of the customer segments
│
├── README.md                         # Documentation
├── requirements.txt                  # Python dependencies
└── segmentation.py                   # Production script to run the full pipeline
```

## How to Run the Project

1. **Activate your environment**:
   Make sure you have your virtual environment activated, or use the appropriate Python path:
   ```bash
   ..\venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. **Run the script**:
   ```bash
   ..\venv\Scripts\python.exe segmentation.py
   ```
   This will read the data from `data/online_retail_II.xlsx`, process it, perform clustering, and save all outputs to the `outputs/` folder.

3. **Or run the notebook**:
   Open `notebooks/Customer_Segmentation.ipynb` using Jupyter Notebook or your favorite editor (like VS Code) and execute the cells.

## Customer Segments Profile (K=4)

Based on the K-Means clustering results, we identified 4 distinct customer segments:

| Cluster | Recency (Avg Days) | Frequency (Avg Purchases) | Monetary (Avg Spend) | Segment Name & Description |
| :---: | :---: | :---: | :---: | :--- |
| **Cluster 0** | ~18 days | ~36.6 | ~$30,598.39 | **VIP/Best Customers**: High frequency, extremely high spending, and recently purchased. |
| **Cluster 1** | ~49 days | ~2.8 | ~$896.56 | **Recent / Low-Spend Customers**: Recently active, but buy infrequently and spend small amounts. |
| **Cluster 2** | ~247 days | ~1.6 | ~$506.20 | **Churned / Lost Customers**: Have not purchased for a long time, very low frequency, and low spend. |
| **Cluster 3** | ~26 days | ~10.3 | ~$4,193.00 | **Loyal Customers**: Frequent purchases, good spending, and active. |

## Visualizations

The generated visualizations are stored under `outputs/figures/`:
- **Elbow Curve (`elbow_method.png`)**: Shows the optimal number of clusters based on WCSS.
- **Customer Clusters (`customer_clusters.png`)**: A 3D scatter plot of Recency vs. Frequency vs. Monetary value showing the clustered customer groups.
