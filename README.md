# Iris K-Means Clustering

## 📌 Project Overview

This project applies the **K-Means Clustering algorithm** to the famous **Iris dataset**.

K-Means is an **unsupervised machine learning algorithm** used to divide data into different groups (clusters) based on similarity.

The Iris dataset contains measurements of iris flowers, and the goal of this project is to group similar flowers into clusters using their numerical features.

## 📊 Dataset

The Iris dataset contains **150 flower samples** with 4 numerical features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The dataset contains three types of iris flowers:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

For K-Means clustering, the flower species/target column is not used during clustering because K-Means is an **unsupervised learning** algorithm.

## 🎯 Objectives

* Understand K-Means clustering.
* Apply K-Means on the Iris dataset.
* Find suitable number of clusters using the **Elbow Method**.
* Visualize the generated clusters.
* Analyze the clustering results.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## 🔄 Project Workflow

1. Load the Iris dataset.
2. Perform data preprocessing.
3. Select numerical features.
4. Determine the optimal number of clusters using the Elbow Method.
5. Apply K-Means Clustering.
6. Predict cluster labels.
7. Visualize the clusters.
8. Analyze the results.

## 🤖 K-Means Clustering

K-Means works by:

1. Selecting the number of clusters **K**.
2. Randomly selecting cluster centroids.
3. Assigning each data point to the nearest centroid.
4. Recalculating the centroid of each cluster.
5. Repeating the process until the centroids become stable.

## 📈 Elbow Method

The **Elbow Method** is used to find the optimal value of K.

It calculates the **Within-Cluster Sum of Squares (WCSS)** for different values of K.

The point where the decrease in WCSS starts becoming slower is called the **elbow point**. This value is selected as the suitable number of clusters.

For the Iris dataset, **K = 3** is commonly selected because the dataset contains three natural flower groups.

## 📌 Sample Python Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Select features
X = df

# Elbow Method
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

print(df.head())
```

## 📊 Results

The K-Means algorithm divides the Iris dataset into **3 clusters**.

The clusters represent groups of flowers having similar measurements based on:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The clusters can be visualized using scatter plots.

## 📷 Visualization

The project includes:

* Elbow Method graph
* Cluster visualization
* Comparison of different flower measurements

## ✅ Conclusion

K-Means Clustering successfully groups the Iris flowers based on their physical measurements. The **Elbow Method** helps identify an appropriate number of clusters, and **3 clusters** provide a meaningful grouping for this dataset.

This project demonstrates how an **unsupervised machine learning algorithm** can discover patterns and groups in data without using predefined class labels.

## 👨‍💻 Author
Varhadi Shreya 
