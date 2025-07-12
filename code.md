import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d


%matplotlib inline
random.seed(19771586)
df = pd.read_csv("musicData.csv")
music_genre = df["music_genre"].copy()
genres_encountered = music_genre.unique()
genres_encountered = genres_encountered[pd.notna(genres_encountered)]
for i in range(len(genres_encountered)):
    music_genre[music_genre == genres_encountered[i]] = i
df["music_genre"] = music_genre

keys = df["key"].copy()
keys_encountered = keys.unique()
keys_encountered = keys_encountered[pd.notna(keys_encountered)]
for i in range(len(keys_encountered)):
    keys[keys == keys_encountered[i]] = i
df["key"] = keys

df.loc[df["tempo"] == "?", "tempo"] = np.nan
df["tempo"] = df["tempo"].astype(float)
dummies = pd.get_dummies(df["mode"])
df = pd.concat([df, dummies], axis=1)
df = df.drop("mode", axis=1)

#print(df.isna().sum()), 5 rows were completely empty
df.dropna(subset=['popularity'], inplace=True)
df = df.drop(["artist_name", "track_name", "instance_id", "obtained_date"], axis=1) #dropped these three columns because they don't have anything to do with genre
actual_output = df["music_genre"]
predictor_data = df.drop(["music_genre"], axis=1)

xTrain, xTest, yTrain, yTest = train_test_split(predictor_data[0:actual_output[actual_output == 0].last_valid_index() + 1], actual_output[0:actual_output[actual_output == 0].last_valid_index() + 1], test_size = 0.1)
last_idx = np.where(actual_output == 0)[0][-1] + 1
for i in range(1, 10):
    x_train_subset, x_test_subset, y_train_subset, y_test_subset = train_test_split(predictor_data[last_idx:actual_output[actual_output == i].last_valid_index() + 1], actual_output[last_idx:actual_output[actual_output == i].last_valid_index() + 1], test_size = 0.1)
    xTrain = pd.concat([xTrain, x_train_subset], ignore_index=True)
    xTest = pd.concat([xTest, x_test_subset], ignore_index=True)
    yTrain = pd.concat([yTrain, y_train_subset], ignore_index=True)
    yTest = pd.concat([yTest, y_test_subset], ignore_index=True)
    last_idx = actual_output[actual_output == i].last_valid_index() + 1

#filling nan values in numpy array

predictor_data = xTrain.dropna()
print(xTrain.corr()["tempo"])
x_known = predictor_data[["acousticness", "energy", "loudness"]]
y_known = predictor_data["tempo"]
clf = RandomForestRegressor(n_estimators=100, max_samples=0.5, max_features=0.5,bootstrap=True, criterion='squared_error')
clf.fit(x_known, y_known)
x_unknown = xTrain[xTrain["tempo"].isna()][["acousticness", "energy", "loudness"]]
y_unknown = clf.predict(x_unknown)
indices = xTrain[xTrain["tempo"].isna()].index
xTrain.loc[indices, "tempo"] = y_unknown

x_unknown = xTest[xTest["tempo"].isna()][["acousticness", "energy", "loudness"]]
y_unknown = clf.predict(x_unknown)
indices = xTest[xTest["tempo"].isna()].index
xTest.loc[indices, "tempo"] = y_unknown

xTrain = xTrain.astype(float).to_numpy()
xTest = xTest.astype(float).to_numpy()
yTrain = yTrain.astype(float).to_numpy()
yTest = yTest.astype(float).to_numpy()

df = pd.read_csv("musicData.csv")
music_genre = df["music_genre"].copy()
genres_encountered = music_genre.unique()
print(genres_encountered)

#to check whether model has linear or nonlinear relationship (what type of dimensionality reduction to use)
svm = LinearSVC(C = 1e10, dual = False) # Uses the squared-hinge loss function when fitting the model.
svm.fit(xTrain, yTrain)

yPred = svm.predict(xTest)

acc = metrics.accuracy_score(yTest, yPred)
print('Linear SVM validation accuracy = {:0.1f}%'.format(100*acc))

"""
since linear SVM accuracy was 28.5% (low), we will use nonlinear dimensionality reduction 
(avoided using MDS and t-SNE because of time complexity, dataset size is 50k--additionally want to preserve both global and local structure)

chose not to use UMAP since it's sensitive to noise/outliers (along with not having metrics to assess quality of the low-dimensional 
embedding--important since this low dimensional data is going to be used for classification afterwards)

and therefore went with autoencoders
"""

scaler = MinMaxScaler()
xTrain[:, 0:12] = scaler.fit_transform(xTrain[:, 0:12])
xTest[:, 0:12] = scaler.transform(xTest[:, 0:12])
#testing type of activation, number of layers, bottleneck (acceptable dimensionality reduction)
class DeepAutoencoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(14, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 14),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DeepAutoencoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(14, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 6)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(6, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 14),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DeepAutoencoder3(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(14, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 14),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DeepAutoencoder4(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(14, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 6),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(6, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 14),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DeepAutoencoder5(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(14, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 3),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 14),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DeepAutoencoder6(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(14, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 6),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(6, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 14),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model1 = DeepAutoencoder1()
model2 = DeepAutoencoder2()
model3 = DeepAutoencoder3()
model4 = DeepAutoencoder4()
model5 = DeepAutoencoder5()
model6 = DeepAutoencoder6()

xTrainTensor = torch.from_numpy(xTrain.astype(np.float32))
xTestTensor = torch.from_numpy(xTest.astype(np.float32))

models = [model1, model2, model3, model4, model5, model6]
train_loader = torch.utils.data.DataLoader(xTrainTensor, batch_size=128)
test_loader = torch.utils.data.DataLoader(xTestTensor, batch_size=128)

for model in models:
    train_loss = []
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # built-in L2
    for t in range(200):
        running_loss = 0
        for batch in train_loader:
            xPred = model(batch)
            loss = criterion(xPred, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(train_loader)

        train_loss.append(running_loss)
        if t == 199:
            plt.scatter([i for i in range(200)], train_loss)
            plt.xlabel('Epoch #')
            plt.ylabel('Train Loss')
            plt.show()
            print([train_loss[i] for i in [49, 99, 199]])

            test_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    xPred = model(batch)
                    loss = criterion(xPred, batch)
                    test_loss += loss.item()
                test_loss /= len(test_loader)
                print(test_loss)

#best 3 dimensional model
reduced_data = model1.encoder(xTrainTensor).detach().numpy()
reduced_test_data = model1.encoder(xTestTensor).detach().numpy()

"""
using kmeans/gmm since we already have number of clusters (since data has labels--10 genres) and other mehtods such as DBSCAN are computationally very expensive for a dataset of 45k points (O(45k^2) = 2 billion operations)
limitations are that it produces spherical clusters (not sure if that was the structure present in the original data) and is sensitive to outliers 
test set size is 5k
"""
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

inertia_kmeans_tech = [0, 0]
init_technique_kmeans = ['k-means++', 'random']
for num in range(10):
    for tech in init_technique_kmeans:
        kmeans = KMeans(n_clusters=10, init=tech)
        kmeans.fit_predict(reduced_data)
        inertia_kmeans_tech[init_technique_kmeans.index(tech)] += kmeans.inertia_
inertia_kmeans_tech = [inertia_kmeans_tech[i] / 10 for i in range(len(inertia_kmeans_tech))]
print(inertia_kmeans_tech)
plt.scatter(init_technique_kmeans, inertia_kmeans_tech) #ran 10 times since the initial location of the centroids can greatly affect performance
plt.xlabel('type of initialization')
plt.ylabel('inertia')
plt.show() #random initialization turns out to be the best

"""
results kept changing here so ommitted
num_init = [10, 20, 30, 40, 50, 60]
inertia_kmeans_init = [0, 0, 0, 0, 0, 0]
for num in range(10):
    for init in num_init:
        kmeans = KMeans(n_clusters=10, init="random", n_init=init)
        kmeans.fit(reduced_data)
        inertia_kmeans_init[num_init.index(init)] += kmeans.inertia_

inertia_kmeans_init = [inertia_kmeans_init[i] / 10 for i in range(len(inertia_kmeans_init))]
print(inertia_kmeans_init)
plt.scatter(num_init, inertia_kmeans_init)
plt.xlabel('# initializations')
plt.ylabel('inertia')
plt.show() #inertia happens to be lowest at 50 initializations
"""
bic_gmm_tech = [0, 0, 0, 0]
init_technique_bic = ['kmeans', 'k-means++', 'random', 'random_from_data']
for num in range(10):
    for tech in init_technique_bic:
        gmm = GaussianMixture(n_components=10, covariance_type = 'full', init_params=tech)
        gmm.fit(reduced_data)
        bic_gmm_tech[init_technique_bic.index(tech)] += gmm.bic(reduced_data)

bic_gmm_tech = [bic_gmm_tech[i] / 10 for i in range(len(bic_gmm_tech))]
print(bic_gmm_tech)
plt.scatter(init_technique_bic, bic_gmm_tech)
plt.xlabel('type of initialization')
plt.ylabel('BIC')
plt.show() #kmeans has best initialization

covariance_types = ['full', 'tied', 'diag', 'spherical']
bic_gmm_cov = [0, 0, 0, 0]

for num in range(10):
    for cov in covariance_types:
        gmm = GaussianMixture(n_components=10, covariance_type = cov, init_params="kmeans")
        gmm.fit(reduced_data)
        bic_gmm_cov[covariance_types.index(cov)] += gmm.bic(reduced_data)

bic_gmm_cov = [bic_gmm_cov[i] / 10 for i in range(len(bic_gmm_cov))]
print(bic_gmm_cov)
plt.scatter(covariance_types, bic_gmm_cov)
plt.xlabel('Covariance Types')
plt.ylabel('BIC')
plt.show()

#compare silhouette scores of the best models
technique = ["kmeans", "gmm"]
silhouette_scores = [0, 0]
for num in range(10):
    kmeans = KMeans(n_clusters=10, init="random")
    gmm = GaussianMixture(n_components=10, covariance_type = "full", init_params="kmeans")
    kmeans.fit(reduced_data)
    gmm.fit(reduced_data)
    silhouette_scores[0] += silhouette_score(reduced_data, kmeans.predict(reduced_data))
    silhouette_scores[1] += silhouette_score(reduced_data, gmm.predict(reduced_data))
silhouette_scores = [silhouette_scores[i] / 10 for i in range(len(silhouette_scores))]
plt.scatter(technique, silhouette_scores)
plt.xlabel('Technique')
plt.ylabel('Silhouette Scores')
plt.show()
labels = []
#kmeans ends up being the best clustering
cluster_labels = kmeans.labels_.copy()

#visualizing dimensionally reduced data
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
cluster_colored = [colors[label] for label in kmeans.labels_]
scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=cluster_colored)
ax.set_xlabel('Autoencoder x values')
ax.set_ylabel('Autoencoder y values')
ax.set_zlabel('Autoencoder z values')
ax.legend()
plt.show()

reduced_data = torch.from_numpy(reduced_data)
reduced_test_data = torch.from_numpy(reduced_test_data)
yTrain = yTrain.long()
yTest = yTest.long()

"""
A neural network was used for a few reasons:
1. After dimensionality reduction via autoencoders, the dimensions that were extracted aren't easily interpretable. With lower interpretability on the data's dimensions,
the understandability of the splits in a decision tree (arguably its most important advantage) is also reduced (regardless of whether it's in a random forest or there are boosting/bagging methods being utilized.)
2. Additionally, data is not a limitation here but SVMs in particular don't natively handle multiclass classification. Making it handle such classification would add 
more training complexity. 

In addition to the 3-dimensional data (reduced), features through clustering labels will be added. Clustering labels help reveal structural relationships in the data.
"""
dimensions = 3
hidden_units1 = 150
hidden_units2 = 200
hidden_units3 = 250
num_classes = 10
learning_rate = 1e-3
lambda_l2 = 1e-3
network1 = nn.Sequential(
    nn.Linear(dimensions, hidden_units1),
    nn.Linear(hidden_units1, num_classes))

network2 = nn.Sequential(
    nn.Linear(dimensions, hidden_units1),
    nn.ReLU(),
    nn.Linear(hidden_units1, num_classes))

network3 = nn.Sequential(
    nn.Linear(dimensions, hidden_units1),
    nn.Sigmoid(),
    nn.Linear(hidden_units1, num_classes))

network4 = nn.Sequential(
    nn.Linear(dimensions, hidden_units2),
    nn.Linear(hidden_units2, num_classes))

network5 = nn.Sequential(
    nn.Linear(dimensions, hidden_units2),
    nn.ReLU(),
    nn.Linear(hidden_units2, num_classes))

network6 = nn.Sequential(
    nn.Linear(dimensions, hidden_units2),
    nn.Sigmoid(),
    nn.Linear(hidden_units2, num_classes))

network7 = nn.Sequential(
    nn.Linear(dimensions, hidden_units3),
    nn.Linear(hidden_units3, num_classes))

network8 = nn.Sequential(
    nn.Linear(dimensions, hidden_units3),
    nn.ReLU(),
    nn.Linear(hidden_units3, num_classes))

network9 = nn.Sequential(
    nn.Linear(dimensions, hidden_units3),
    nn.Sigmoid(),
    nn.Linear(hidden_units3, num_classes))
list_of_neural_networks = [network1, network2, network3, network4, network5, network6, network7, network8, network9]
for model in list_of_neural_networks:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lambda_l2) # built-in L2
    for t in range(1000):
        yPred = model(reduced_data)
        loss = criterion(yPred, yTrain)
        score, predicted = torch.max(yPred, 1)
        acc = (yTrain == predicted).sum() / len(yTrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t == 999:
            print("[LOSS]: %.6f, [ACCURACY]: %.3f" % (loss.item(), acc))
    out = model(reduced_test_data)
    _, predicted = torch.max(out.data, 1)
    print('Test Accuracy of the network %.4f %%' % (100 * torch.sum(yTest==predicted).double() / len(yTest)))

learning_rate = 1e-2
network1 = nn.Sequential(
    nn.Linear(dimensions, hidden_units1),
    nn.Linear(hidden_units1, num_classes))

network2 = nn.Sequential(
    nn.Linear(dimensions, hidden_units1),
    nn.ReLU(),
    nn.Linear(hidden_units1, num_classes))

network3 = nn.Sequential(
    nn.Linear(dimensions, hidden_units1),
    nn.Sigmoid(),
    nn.Linear(hidden_units1, num_classes))

network4 = nn.Sequential(
    nn.Linear(dimensions, hidden_units2),
    nn.Linear(hidden_units2, num_classes))

network5 = nn.Sequential(
    nn.Linear(dimensions, hidden_units2),
    nn.ReLU(),
    nn.Linear(hidden_units2, num_classes))

network6 = nn.Sequential(
    nn.Linear(dimensions, hidden_units2),
    nn.Sigmoid(),
    nn.Linear(hidden_units2, num_classes))

network7 = nn.Sequential(
    nn.Linear(dimensions, hidden_units3),
    nn.Linear(hidden_units3, num_classes))

network8 = nn.Sequential(
    nn.Linear(dimensions, hidden_units3),
    nn.ReLU(),
    nn.Linear(hidden_units3, num_classes))

network9 = nn.Sequential(
    nn.Linear(dimensions, hidden_units3),
    nn.Sigmoid(),
    nn.Linear(hidden_units3, num_classes))
list_of_neural_networks = [network1, network2, network3, network4, network5, network6, network7, network8, network9]

for model in list_of_neural_networks:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lambda_l2) # built-in L2
    for t in range(1000):
        yPred = model(reduced_data)
        loss = criterion(yPred, yTrain)
        score, predicted = torch.max(yPred, 1)
        acc = (yTrain == predicted).sum() / len(yTrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t == 999:
            print("[LOSS]: %.6f, [ACCURACY]: %.3f" % (loss.item(), acc))
    out = model(reduced_test_data)
    _, predicted = torch.max(out.data, 1)
    print('Test Accuracy of the network %.4f %%' % (100 * torch.sum(yTest==predicted).double() / len(yTest)))

"""
trying with multiple layers
3 observations (related to performance on both training and test sets): 
1. Best with linear models (no activation)
2. Better with higher learning rate --> avoiding get stuck at local minimas
3. Improves with more layers but only for linear model

Indicates that the data is linearly separable but the hyperplanes may not be simple (equation wise)--further supported by the fact that a 
higher number of hidden units is leading to better performance  
"""
hidden_units = 150
linear_network1 = nn.Sequential(
    nn.Linear(dimensions, hidden_units),
    nn.Linear(hidden_units, hidden_units),
    nn.Linear(hidden_units, num_classes))

relu_network1 = nn.Sequential(
    nn.Linear(dimensions, hidden_units),
    nn.ReLU(),
    nn.Linear(hidden_units, hidden_units),
    nn.ReLU(),
    nn.Linear(hidden_units, num_classes))

sigmoid_network1 = nn.Sequential(
    nn.Linear(dimensions, hidden_units),
    nn.Sigmoid(),
    nn.Linear(hidden_units, hidden_units),
    nn.Sigmoid(),
    nn.Linear(hidden_units, num_classes))

hidden_units = 250
linear_network2 = nn.Sequential(
    nn.Linear(dimensions, hidden_units),
    nn.Linear(hidden_units, hidden_units),
    nn.Linear(hidden_units, num_classes))

relu_network2 = nn.Sequential(
    nn.Linear(dimensions, hidden_units),
    nn.ReLU(),
    nn.Linear(hidden_units, hidden_units),
    nn.ReLU(),
    nn.Linear(hidden_units, num_classes))

sigmoid_network2 = nn.Sequential(
    nn.Linear(dimensions, hidden_units),
    nn.Sigmoid(),
    nn.Linear(hidden_units, hidden_units),
    nn.Sigmoid(),
    nn.Linear(hidden_units, num_classes))

list_of_neural_networks = [linear_network1, relu_network1, sigmoid_network1, linear_network2, relu_network2, sigmoid_network2]

for model in list_of_neural_networks:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lambda_l2) # built-in L2
    for t in range(1000):
        yPred = model(reduced_data)
        loss = criterion(yPred, yTrain)
        score, predicted = torch.max(yPred, 1)
        acc = (yTrain == predicted).sum() / len(yTrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t == 999:
            print("[LOSS]: %.6f, [ACCURACY]: %.3f" % (loss.item(), acc))
    out = model(reduced_test_data)
    _, predicted = torch.max(out.data, 1)
    print('Test Accuracy of the network %.4f %%' % (100 * torch.sum(yTest==predicted).double() / len(yTest)))

#for the reasons stated above, we will use linearSVC as our final classifier--not great justification, but using the 3D graph created via kmeans, the clusters also look linearly separable (which may further support this informally)
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from scipy.special import softmax
#best 3 dimensional model
reduced_data = model1.encoder(xTrainTensor).detach().numpy()
reduced_test_data = model1.encoder(xTestTensor).detach().numpy()

one_hot1 = np.zeros((cluster_labels.size, 10))
one_hot1[np.arange(cluster_labels.size), cluster_labels] = 1
reduced_data = np.hstack((reduced_data, one_hot1))

test_predict = kmeans.predict(reduced_test_data)
one_hot2 = np.zeros((test_predict.size, 10))
one_hot2[np.arange(test_predict.size), test_predict] = 1
reduced_test_data = np.hstack((reduced_test_data, one_hot2))

#model has better performance w/l2
svm = LinearSVC(penalty="l1", max_iter=2000, dual=False)
svm.fit(reduced_data, yTrain)
decision_probabilities = softmax(svm.decision_function(reduced_test_data), axis = 1)
print("AUC:", roc_auc_score(yTest, decision_probabilities, multi_class='ovr'))

svm = LinearSVC(max_iter=2000, dual=False)
svm.fit(reduced_data, yTrain)
decision_probabilities = softmax(svm.decision_function(reduced_test_data), axis = 1)
print("AUC:", roc_auc_score(yTest, decision_probabilities, multi_class='ovr'))

print("AUC:", roc_auc_score(yTest, decision_probabilities, multi_class='ovr', average=None))

regularization = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10] #regularization of 10 seems to be the best performer
for c in regularization:
    svm = LinearSVC(C=c, max_iter=2000, dual=False)
    svm.fit(reduced_data, yTrain)
    decision_probabilities = softmax(svm.decision_function(reduced_test_data), axis = 1)
    print("AUC:", roc_auc_score(yTest, decision_probabilities, multi_class='ovr'))
    if c == 10:
        print("AUC:", roc_auc_score(yTest, decision_probabilities, multi_class='ovr', average=None))
        pred = np.argmax(decision_probabilities, axis=1)
        conf_matrix = metrics.confusion_matrix(yTest, pred)
        print("Confusion matrix = ")
        print(conf_matrix)
        all_precision = metrics.precision_score(yTest, pred, average=None)
        all_recall = metrics.recall_score(yTest, pred, average=None)
        all_f1 = metrics.f1_score(yTest, pred, average=None)

        for idx in range(len(all_precision)):
            precision = all_precision[idx]
            recall = all_recall[idx]
            f1_score = all_f1[idx]
            print(f"Precision = {precision}, Recall = {recall}, f1 score = {f1_score}")

        to_graph = yTest.numpy()
        for cl in range(10):
            y_true_binary = (to_graph == cl).astype(int)
            y_score = decision_probabilities[:, cl]
            print(f"Genre #{cl+1}")
            RocCurveDisplay.from_predictions((to_graph == cl).astype(int), decision_probabilities[:, cl])

plt.show()
"""
#changing loss function, much worse
svm = LinearSVC(C=2, loss="hinge", dual=True) 
svm.fit(reduced_data, yTrain)
decision_probabilities = softmax(svm.decision_function(reduced_test_data), axis = 1)
print("AUC:", roc_auc_score(yTest, decision_probabilities, multi_class='ovr'))

#changing max number of iterations--didn't make a difference at all
iterations = [500, 1000, 1500, 2000, 2500]
for itr in iterations:
    svm = LinearSVC(C=2, max_iter=itr, dual=False) 
    svm.fit(reduced_data, yTrain)
    decision_probabilities = softmax(svm.decision_function(reduced_test_data), axis = 1)
    print("AUC:", roc_auc_score(yTest, decision_probabilities, multi_class='ovr'))
"""
