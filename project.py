import csv
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = []
with open('data.csv', 'r') as f:
    file = csv.reader(f)
    for row in file:
        df.append(row)

headers = df[0]
headers[0] = "row_num"
star_data = df[1:]
# print(star_data)

planet_masses = []
planet_radiuses = []

for data in star_data:
    planet_masses.append(data[3])
    planet_radiuses.append(data[4])
# print(planet_masses)

X = []

for index, planet_mass in enumerate(planet_masses):
    temp_list = [
        planet_radiuses[index],
        planet_mass
    ]
    X.append(temp_list)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 42)
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()





# Plotting the clusters
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

cluster_1_x = []
cluster_1_y = []
cluster_2_x = []
cluster_2_y = []
cluster_3_x = []
cluster_3_y = []
cluster_4_x = []
cluster_4_y = []

for index, data in enumerate(X):
  if y_kmeans[index] == 0:
    cluster_1_x.append(data[0])
    cluster_1_y.append(data[1])
  elif y_kmeans[index] == 1:
    cluster_2_x.append(data[0])
    cluster_2_y.append(data[1])
  elif y_kmeans[index] == 2:
    cluster_3_x.append(data[0])
    cluster_3_y.append(data[1])
  elif y_kmeans[index] == 3:
    cluster_4_x.append(data[0])
    cluster_4_y.append(data[1])

#Chart with Scatter Plot
plt.figure(figsize=(15,7))
sns.scatterplot(cluster_1_x, cluster_1_y, color = 'yellow', label = 'Cluster 1')
sns.scatterplot(cluster_2_x, cluster_2_y, color = 'blue', label = 'Cluster 2')
sns.scatterplot(cluster_3_x, cluster_3_y, color = 'green', label = 'Cluster 3')
sns.scatterplot(cluster_4_x, cluster_4_y, color = 'red', label = 'Cluster 4')
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'black', label = 'Centroids',s=100,marker=',')
plt.title('Clusters of Planets')
plt.xlabel('Planet Radius')
plt.ylabel('Planet Mass')
plt.legend()
plt.gca().invert_yaxis()
plt.show()