from kmodes.kmodes import KModes #a lot of Kmodes in here
from ucimlrepo import fetch_ucirepo #the dataset I will be using

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


#I will be using a mushroom dataset, where we have data about a mushroom and want to decide whether it's poisonous or not
#To visualize, I will be taking 2 features out of the many features it has "cap-shape" and "gill-color"

dataset = fetch_ucirepo(id=73)

x = dataset.data.features[["cap-shape", "gill-color"]]

#suppose we don't know this
y = dataset.data.targets

#I copy pasted the following code from 2.py (the k-means code)

k_modes_models = []

#we will have clusters of up to 10
up_to_cluster_size = 10

#this variable is used at the end, when we are about to plot the elbow plot and accuracy plot
#it contains integers, which is the number of clusters
k_values = []

#this is used for plotting the elbow plot, contains wcss values for each k-mean
wcss = []

for i in range(up_to_cluster_size):
    k_means_model = KModes(n_clusters=i + 1)
    k_means_model.fit(x)
    k_modes_models.append(k_means_model)

    k_values.append(i + 1)
    #for the elbow plot, there is a nice attribute "inertia_" that has the WCSS already calculated for us, we just use that
    wcss.append(k_means_model.cost_)

figure, sub_plt = plt.subplots(2, 3, figsize=(8, 7))

#python starts plotting with 2 clusters up to 2 + 3 = 5, if you want to change it (lets say from 5 to 9) you can change this number to 5
starting_clusters = 3

#some mix of my code and online code to draw 4 different plots
for i in range(5):
    sub_plt[i // 3][i % 3].scatter(x["cap-shape"], x["gill-color"], s=35)
    sub_plt[i // 3][i % 3].scatter(k_modes_models[i + starting_clusters - 1].cluster_centroids_[:, 0], k_modes_models[i + starting_clusters - 1].cluster_centroids_[:, 1], s=250, c='red', marker='X')  # Cluster centers
    sub_plt[i // 3][i % 3].set_title(f"K-means with {i + starting_clusters} clusters")
    sub_plt[i // 3][i % 3].set_xlabel("cap shape")
    sub_plt[i // 3][i % 3].set_ylabel("gill color")



#plot the elbow curve in the first row's last column
sub_plt[1][2].plot(k_values, wcss)
sub_plt[1][2].set_title("elbow plot")
sub_plt[1][2].set_xticks(k_values)

plt.tight_layout()
plt.show()

#Interestingly, as we increase k, it seems that the chosen centroids are the exact same, with 1 more extra centroid being added