import matplotlib.pyplot as plt
import numpy as np
import math
import operator
'''
instance1 : 28*28 1 d array
instance2: 28*28 1 d array
len: length of array i.e 28*28
'''
def euclidian_distance(instance1,instance2):
    return math.sqrt(np.sum(np.square(instance1 - instance2)))
'''
Takes n-2D arrays of trainingData
      test_labels per training_labels
      trainingSample to be tested
'''
def getneighbours(trainingData,training_labels,testSample,k):
    distances = []
    neighbours = []
    for index,trainingInstance in enumerate(trainingData):
        #length = len(testInstance.flatten())
        dist = euclidian_distance(trainingInstance,testSample)
        distances.append((training_labels[index],dist))
    distances.sort(key=operator.itemgetter(1))
    for i in range(k):
        neighbours.append(distances[i][0])
    #print neighbours
    return neighbours
'''
Takes neighbours and find who occur the most
'''
def getmostoccuringlabel(neighbours):
    votes = {}
    for neighbour in neighbours:
        if neighbour in votes:
            votes[neighbour] +=1
        else:
            votes[neighbour] =1
    votes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
    return votes[0][0]
'''
Get the list of labels
'''
def label_list(filename):
    fileObj = open(filename,'rb')
    dataRead = fileObj.read()
    dataReadFromB16 = dataRead[8:]
    labels = np.frombuffer(dataReadFromB16, dtype=np.ubyte)
    return labels

'''
Get the data matrix of the data to be either trained or tested
'''
def data_sorting(filename,size):
    fileObj2 = open(filename, "rb")
    dataRead = fileObj2.read()
    dataReadFromB16 = dataRead[16:]
    image1DArray = np.frombuffer(dataReadFromB16, dtype=np.ubyte)
    matrix_2 = image1DArray.reshape((size * 28, 28))
    matrix_3 = matrix_2.reshape((size, 28, 28))
    matrix_3 = matrix_3 / 255.0 ##Noramlizing value to 0-1
    return matrix_3


if __name__ == "__main__":


    #read Train Label

    train_x_file = "./train-images.idx3-ubyte"
    train_matrix = data_sorting(train_x_file, 60000)
    print (train_matrix.shape)

    train_y_file = "./train-labels.idx1-ubyte"
    train_label = label_list(train_y_file)
    print(train_label.shape)

    train_x_file = "./t10k-images.idx3-ubyte"
    test_matrix = data_sorting(train_x_file, 10000)
    print(test_matrix.shape)

    train_y_file = "./t10k-labels.idx1-ubyte"
    test_label = label_list(train_y_file)
    print(test_label.shape)
    predictions =[]
    k_values = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]
    accuracy = [0.0]*len(k_values)
    for ind,k in enumerate(k_values):
        print ("For K: "+str(k))

        total_count = len(test_matrix)
        correct_count = 0
        for x in range(len(test_matrix)):
            neighbors = getneighbours(train_matrix, train_label, test_matrix[x], k)
            result = getmostoccuringlabel(neighbors)
            predictions.append(result)
            #print (x)
            #print('> predicted=' + repr(result) +', actual=' + repr(test_label[x]))
            if result == test_label[x]:
                correct_count +=1
        accuracy[ind] =((float(correct_count)/total_count)*100.0)
    print (k_values)
    print (accuracy)
    k_values = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]
    #accuracy = [100, 10, 20, 10, 10, 29, 67, 56, 67, 19]
    plt.plot(k_values, accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('k values')
    plt.savefig('./KNN.png')