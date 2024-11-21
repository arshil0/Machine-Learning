#sklearn provides a nice function "make_blobs" that generates a nice set of data that are blobs and could be clustered
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d

#number of centers we want to have in our random data
n_centers = 4

# Generate blobs (suppose we don't have y to immitate having unsupervised data)
#there will be 4 clusters of blobs with a standard deviation of 0.65 to make it a bit more challenging for the model
x, y = make_blobs(n_samples=300, centers=n_centers, cluster_std=0.65)

#I will create an array of K-means clusters, the index + 1 is the number of clusters.
#So the K-means at index 0 will have 1 cluster.
#K-means at index 1 will have 2 clusters, etc...

k_means_models = []

#we will have clusters of up to 10
up_to_cluster_size = 10

#this variable is used at the end, when we are about to plot the elbow plot and accuracy plot
#it contains integers, which is the number of clusters
k_values = []

#this is used for plotting the elbow plot, contains wcss values for each k-mean
wcss = []

#this is the accuracy score for each k-means clustering
accuracies = []

for i in range(up_to_cluster_size):
    k_means_model = KMeans(n_clusters=i + 1)
    k_means_model.fit(x)
    k_means_models.append(k_means_model)

    k_values.append(i + 1)
    #for the elbow plot, there is a nice attribute "inertia_" that has the WCSS already calculated for us, we just use that
    wcss.append(k_means_model.inertia_)

    #calculate the accuracy of each model, and see how well it actually did
    #I know this is supposed to be unsupervised, but it's interesting to see the accuracies
    y_pred = k_means_model.predict(x)

    #however, there is a problem, the predicted values may be off but correct (so 2 is always predicted as 1 for example)
    #to fix this, we need to map the values to their corresponding values
    
    #an array of arrays (where each sub array has 2 numbers, the first number is the original value, the second is the value to change to)
    changes_to_be_made = []

    #pretty weird code but it works (hopefully) :)
    #goes through each of the different values up to the number of centers we have and fixes their values
    for f in range(min(i + 1, n_centers)):
        value_to_change_to = 0
        for index, value in enumerate(y_pred):
            if value == f:
                value_to_change_to = y[index]
        changes_to_be_made.append([f, value_to_change_to])

    for index, value in enumerate(y_pred):
        for change_list in changes_to_be_made:
            if value == change_list[0]:
                y_pred[index] = change_list[1]

    accuracies.append(accuracy_score(y_pred, y))


#I want to plot different k-means plots and then plot the elbow curve
figure, sub_plt = plt.subplots(2, 3, figsize=(8, 7))

#plot the blobs and the centroids for each k-means clustering from 2 to 5

#python starts plotting with 2 clusters up to 2 + 3 = 5, if you want to change it (lets say from 5 to 9) you can change this number to 5
starting_clusters = 2

#some mix of my code and online code to draw 4 different plots
for i in range(4):
    sub_plt[i // 2][i % 2].scatter(x[:, 0], x[:, 1], s=35)
    sub_plt[i // 2][i % 2].scatter(k_means_models[i + starting_clusters - 1].cluster_centers_[:, 0], k_means_models[i + starting_clusters - 1].cluster_centers_[:, 1], s=250, c='red', marker='X')  # Cluster centers
    sub_plt[i // 2][i % 2].set_title(f"K-means with {i + starting_clusters} clusters")



#plot the elbow curve in the first row's last column
sub_plt[0][2].plot(k_values, wcss)
sub_plt[0][2].set_title("elbow plot")
sub_plt[0][2].set_xticks(k_values)

#plot the accuracy of each k-means clustering in the last column of the second row
sub_plt[1][2].plot(k_values, accuracies)
sub_plt[1][2].set_title("Accuracies")
sub_plt[1][2].set_xlabel("cluster size")
sub_plt[1][2].set_xticks(k_values)

plt.tight_layout()
plt.show()