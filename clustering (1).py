import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Retail Customer Segmentation – Clustering")

# Load data
df = pd.read_csv("retail_sales_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])

# RFM
snapshot_date = df['Date'].max() + pd.Timedelta(days=1)
rfm = df.groupby('Customer ID').agg({
    'Date': lambda x: (snapshot_date - x.max()).days,
    'Transaction ID': 'count',
    'Total Amount': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Scaling
X = rfm[['Recency','Frequency','Monetary']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Hierarchical
linked = linkage(X_scaled, method='ward')
rfm['Hierarchical_Cluster'] = fcluster(linked, t=4, criterion='maxclust')

# DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
rfm['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
rfm['PCA1'] = X_pca[:,0]
rfm['PCA2'] = X_pca[:,1]

# Sidebar
method = st.sidebar.selectbox("Select Clustering Method", ["KMeans","Hierarchical","DBSCAN"])

st.subheader("Clustered Customer Data")
st.dataframe(rfm.head())

# Plot
st.subheader("Cluster Visualization (PCA)")
fig, ax = plt.subplots()
if method == "KMeans":
    ax.scatter(rfm['PCA1'], rfm['PCA2'], c=rfm['KMeans_Cluster'], cmap='tab10')
elif method == "Hierarchical":
    ax.scatter(rfm['PCA1'], rfm['PCA2'], c=rfm['Hierarchical_Cluster'], cmap='tab10')
else:
    ax.scatter(rfm['PCA1'], rfm['PCA2'], c=rfm['DBSCAN_Cluster'], cmap='tab10')

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
st.pyplot(fig)

# Cluster Profiling
st.subheader("Cluster Profiling (Mean RFM)")
if method == "KMeans":
    st.dataframe(rfm.groupby('KMeans_Cluster')[['Recency','Frequency','Monetary']].mean())
elif method == "Hierarchical":
    st.dataframe(rfm.groupby('Hierarchical_Cluster')[['Recency','Frequency','Monetary']].mean())
else:
    st.dataframe(rfm[rfm['DBSCAN_Cluster'] != -1].groupby('DBSCAN_Cluster')[['Recency','Frequency','Monetary']].mean())
