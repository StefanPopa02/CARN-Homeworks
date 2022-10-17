import numpy as np
import matplotlib.pyplot as plt


def visualize(a, b, c, x1, y1, x2, y2):
    x = np.linspace(0, 100, 100)
    y = (a * x + c) / (-1 * b)
    plt.plot(x, y, '-r')
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('y', color='#1C2833')
    plt.grid()

    plt.scatter(x1, y1, edgecolors='red')
    plt.scatter(x2, y2, edgecolors='pink')
    plt.show()


def isToTheLeftOfLine(x_point, y_point, indices):
    return (indices[0] * x_point + indices[1] * y_point + indices[2]) > 0


def classifyPoint(x, y, label, indices):
    leftSide = isToTheLeftOfLine(x, y, indices)
    if label == -1 and leftSide:
        return True
    elif label == 1 and not leftSide:
        return True
    else:
        return False


def hillClimb(allPoints, counter, globalCounter, globalIndices, indices, stillSearching):
    for point in allPoints:
        x_point = point[0]
        y_point = point[1]
        label_point = point[2]
        classification = classifyPoint(x_point, y_point, label_point, indices)
        if classification:
            counter = counter + 1
        print("x=", point[0], " y=", point[1], " label=", point[2], "classif", classification)
    if counter > globalCounter:
        stillSearching = True
        globalCounter = counter
        globalIndices = indices
    return globalCounter, globalIndices, stillSearching


def solve(allPoints):
    stillSearching = True
    globalCounter = 0
    globalIndices = np.array([-1000, 1, 47000])  # one optimal solution
    # globalIndices = np.array([1, -1, 0])  # initialize a=1, b=-1, c=0 resulting y=x
    while stillSearching:
        stillSearching = False
        for offset in np.array([1, -1]):
            for indicesIdx in range(2):
                counter = 0
                indices = globalIndices.copy()
                if indices[1] + offset == 0:  # b != 0 for plotting
                    continue
                indices[indicesIdx] += offset
                globalCounter, globalIndices, stillSearching = hillClimb(allPoints, counter, globalCounter,
                                                                         globalIndices, indices, stillSearching)
    print("Correct classified:", globalCounter)
    print("Indices", globalIndices)
    return globalIndices


x1 = np.random.uniform(0, 45, 50)
y1 = np.random.uniform(0, 100, 50)
leftPoints = np.vstack((x1, y1, np.ones(50) * -1)).T

x2 = np.random.uniform(55, 100, 50)
y2 = np.random.uniform(0, 100, 50)
rightPoints = np.vstack((x2, y2, np.ones(50))).T

allPoints = np.concatenate((leftPoints, rightPoints))
np.random.shuffle(allPoints)
print("Number of points:", len(allPoints))

a = 1
b = -1
c = 0
indices = solve(allPoints)
visualize(indices[0], indices[1], indices[2], x1, y1, x2, y2)
