import copy
from faker import Faker
import heapq
import pandas as pd
import scipy

fake = Faker()

# Get metadata from a graph
# - nodes: all the nodes in a graph
# - deg: degree of a node
def metadata(ug):
    nodes = []
    deg = {}
    for e in ug:
        nodes.append(e)
        deg[e] = len(ug[e])
    return nodes, deg

# Input:
# - graph
#     An directed graph represent as a map of vertices
#     with each entry represent an edge
# Output:
# - seedNodes
#     A list of nodes that can be used as seed for community
#     structure labeling
def findSeedNodes(ug):
    copiedUG = copy.deepcopy(ug)
    candidates, deg = metadata(copiedUG)
    seedNodes = []
    while len(candidates) != 0:
        highestDegreeNode = max(deg, key=deg.get)
        seedNodes.append(highestDegreeNode)
        neighbors = copiedUG.pop(highestDegreeNode)
        neighbors.append(highestDegreeNode)
        removeNodes = neighbors
        for removedNode in removeNodes:
            if removedNode in copiedUG:
                copiedUG.pop(removedNode)
            for node in copiedUG:
                if removedNode in copiedUG[node]:
                    copiedUG[node].remove(removedNode)
        candidates, deg = metadata(copiedUG)
    return seedNodes

# Find keys with max value in a dictionary
def findKeysWithMax(d):
    vs = [d[k] for k in d]
    ks = [k for k in d]
    V = max(vs)
    maxKeys = []
    for k in ks:
        if d[k] == V:
            maxKeys.append(k)
    return maxKeys

# Extended Kronecker delta
def extendedKroneckerDelta(ug, node, nodeLabels):
    Nv = ug[node]
    labelRank = {}
    for n in Nv:
        nLabels = nodeLabels[n]
        for l in nLabels:
            if l not in labelRank:
                labelRank[l] = 0
            labelRank[l] = labelRank[l] + 1
 
    if len(labelRank) == 0:
        return nodeLabels

    nodeLabels[node] = findKeysWithMax(labelRank)
    return nodeLabels
    
def findLabelOccupations(nodeLabels):
    labelOccupation = {}
    for n in nodeLabels:
        labels = nodeLabels[n]
        for l in labels:
            if l not in labelOccupation:
                labelOccupation[l] = 0
            labelOccupation[l] = labelOccupation[l]+1
    return labelOccupation


def labelPropagation(ug, labels):
    beforePropagation = {}
    afterPropagation = copy.deepcopy(labels)
    Nvt = {}
    while beforePropagation != afterPropagation:
        beforePropagation = copy.deepcopy(afterPropagation)
        for n in ug:
            afterPropagation = extendedKroneckerDelta(ug, n, afterPropagation)

            # For each t update Nv
            labelOccupations = findLabelOccupations(afterPropagation)
            for l in labelOccupations:
                if l not in Nvt:
                    Nvt[l] = []
                Nvt[l].append(labelOccupations[l])
    return afterPropagation, Nvt

def findLabelCentrality(Nvt):
    labeCentralities = {}
    for l in Nvt:
        labeCentralities[l] = max(Nvt[l])
    return labeCentralities

def assignLabel(nodes):
    nodeLabels = {}
    for n in nodes:
        nodeLabels[n] = [fake.name()]
    return nodeLabels

def matrixFormToArrayForm(ndarr):
    ug = {}
    for r in range(len(ndarr)):
        for c in range(len(ndarr[r])):
            if r not in ug:
                ug[r] = []
            if ndarr[r][c] == 1:
                ug[r].append(c)
    return ug

def readDataSet():
    social_dataset = scipy.io.mmread('socfb-Caltech36.mtx')
    social_dataset_df = pd.DataFrame.sparse.from_spmatrix(social_dataset)
    return matrixFormToArrayForm(social_dataset_df.to_numpy())

def imlpa(ug):
    # Find seed nodes
    seedNodes = findSeedNodes(ug)
    # print("seed nodes", seedNodes)

    # Assign label to seed nodes
    seedNodeLabels = assignLabel(seedNodes)
    # print("seed node labels", seedNodeLabels)

    # Non-seed nodes should be assigned with empty label set
    nodeLabels = copy.deepcopy(seedNodeLabels)
    nodes, deg = metadata(ug)
    for n in nodes:
        if n not in nodeLabels:
            nodeLabels[n] = []

    # Start propagating labels until labels distribution is stable
    propagatedLabels, Nvt = labelPropagation(ug, nodeLabels)
    print(propagatedLabels)

    # Find labels centrality to know which node has the most influence
    labelCetrality = findLabelCentrality(Nvt)
    nc = {}
    for n in seedNodes:
        label = seedNodeLabels[n][0]
        if label not in labelCetrality:
            continue
        nc[n] = labelCetrality[seedNodeLabels[n][0]]

    # Get N most influence nodes
    return propagatedLabels, nc, nodeLabels

def getNMostInfluenceNode(nodeCentralities, nodeLabels, noCelebrities):
    kMostInfluenceNodes = heapq.nlargest(noCelebrities, nodeCentralities.items(), key=lambda i: i[1])
    celebs = {k[0]:nodeLabels[k[0]] for k in kMostInfluenceNodes}
    return celebs

def extractCommnunity(propagatedLabels):
    community = {"none": []}
    for node in propagatedLabels:
        labels = propagatedLabels[node]
        if len(labels) == 0:
            community["none"].append(node)
        else:
            l = labels[0]
            if labels[0] not in community:
                community[l] = []
            community[l].append(node)
    return community

ug = readDataSet()
propagatedLabels, nodeCentralities, nodeLabels = imlpa(ug)
celebs = getNMostInfluenceNode(nodeCentralities, nodeLabels, 3)
print(celebs)

communities = extractCommnunity(propagatedLabels)
print("communities", communities)