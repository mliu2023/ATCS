import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

class ATCS_KMeans:
    # default: 
    # k = 8
    # 10 iterations
    # random initialization of centriods
    # tolerance = .001
    # max number of iterations is 100
    # constructor for the ATCS_KMeans class
    def __init__(self, points, k = 8, nbInit = 10, init = "random", tolerance = .001, maxIter = 100):
        self.points = points
        self.k = k
        self.nbInit = nbInit
        self.init = init
        self.tolerance = tolerance
        self.maxIter = maxIter
        self.centroids = []
        self.classification = [0]*len(points)
        self.centroidClusters = [[] for i in range(k)]

    # initialzies the centroids either randomly or using the KMeans++ method
    def init_centroids_random(self):
        if(self.init == "random"):
            # randomly choose k points
            self.centroids = np.array(self.points[np.random.choice(self.points.shape[0], self.k, replace = False)])
        elif(self.init == "++"):
            # add the first random centroid
            point = self.points[np.random.choice(self.points.shape[0], replace = False)]
            self.centroids.append(point)
            
            # add k-1 centroids by weighting probabilities based on distance from previous centroids
            
            # weighted probabilities
            weights = np.zeros(len(self.points))
            for i in range(1, self.k):
                # update the weights by adding the distance to the previously added centroid
                distances = sp.spatial.distance.cdist(self.points, [self.centroids[i-1]])
                for j in range(0, len(self.points)):
                    weights[j] += distances[j][0]
                self.centroids.append(self.points[np.random.choice(self.points.shape[0], replace = False, p = weights/np.sum(weights))])
            self.centroids = np.array(self.centroids)
                            
    def draw_state(self):
        ax = sns.scatterplot(x = self.points[:,0], y = self.points[:,1], hue = self.classification, palette = 'bright')
        xArr = []
        yArr = []
        print(self.centroids)
        for i in range(0, self.k):
            xArr.append(self.centroids[i][0])
            yArr.append(self.centroids[i][1])
        sns.scatterplot(x = xArr, y = yArr, hue = np.arange(self.k), palette = 'bright', marker = "*", s = 500)
        
        # putting legend labels in a dictionary to remove repeats
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()
        
    def classify_points(self):
        # puts all distances between points and centroids in an array
        distances = sp.spatial.distance.cdist(self.points, self.centroids)
        self.centroidClusters = [[] for i in range (self.k)]
        for i in range(0, len(self.points)):
            minIndex = 0
            minDist = distances[i][0]
            for j in range(0, len(distances[i])):
                # updating the minimum distance from point i to a centroid
                if(distances[i][j] < minDist):
                    minIndex = j
                    minDist = distances[i][j]
            self.classification[i] = minIndex
            self.centroidClusters[minIndex].append(self.points[i])
                
    def calculate_SSE(self):
        # calculates the sum of squared errors
        total = 0
        for i in range(0, len(self.points)):
            distance = sp.spatial.distance.euclidean(self.points[i], self.centroids[self.classification[i]])
            total += distance*distance
        return total
    
    def update_centroids(self):
        # finds the mean of points to find their centroid
        for i in range(0, self.k):
            if(len(self.centroidClusters[i]) > 0):
                self.centroids[i] = np.mean(self.centroidClusters[i], axis = 0)
            
    def cluster_points(self, verbose = False):
        # bestSSE and bestCluster will be assigned meaningful values after the first iteration through the for loop
        bestSSE = -1
        bestCluster = ATCS_KMeans(self.points, self.k, self.nbInit, self.init, self.tolerance, self.maxIter)
        for i in range(0, self.nbInit):
            SSE = 0
            stop = False
            iteration = 0
            cluster = ATCS_KMeans(self.points, self.k, self.nbInit, self.init, self.tolerance, self.maxIter)
            cluster.init_centroids_random()
            if verbose:
                print("Iteration: ", i)
            while not stop and iteration < self.maxIter:
                cluster.classify_points()
                newSSE = cluster.calculate_SSE()
                if verbose:
                    print("SSE:", newSSE)
                cluster.update_centroids()
                if abs(newSSE - SSE) < cluster.tolerance*SSE:
                    stop = True
                SSE = newSSE
                iteration += 1
            if(bestSSE == -1 or SSE < bestSSE):
                bestSSE = SSE
                bestCluster = cluster
         
        self.__dict__.update(bestCluster.__dict__)
        if verbose:
            print("Final SSE:", bestSSE)
            bestCluster.draw_state()
            
    def elbow_graph(self):
        Ks = range(2, 10)
        SSEs = []
        for k in Ks:
            clustering = ATCS_KMeans(self.points, k, self.nbInit, self.init, self.tolerance, self.maxIter)
            clustering.cluster_points(False)
            SSEs.append(clustering.calculate_SSE())
        # Plot the SSE vs the number of cluster centers
        plt.figure()
        plt.plot(Ks, SSEs, marker = "o")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Within Clusters Sum of Square Error")
        plt.savefig("SSEs.jpg")
        plt.show()