import copy
from faker import Faker
import heapq
import pandas as pd
import scipy
import networkx as nx
import matplotlib.pyplot as plt
import random

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
    keys = []
    for key in d:
        vals = d[key]
        for i in range(vals):
            keys.append(key)
    maxKeys = []
    maxKeys.extend(random.choices(keys, k=1))
    # for k in ks:
        # if d[k] == V:
        #     maxKeys.append(k)
    return maxKeys

# Unify distribution of p
def shouldPropagate(p):
    arr = [0]*round(1/p)
    arr[0] = 1
    return random.choice(arr) == 1

# Extended Kronecker delta independant cascade
def extendedKroneckerDelta(ug, node, nodeLabels):
    neighbors = ug[node]
    labelRank = {}
    for neighbor in neighbors:
        neighborLabels = nodeLabels[neighbor]
        if len(nodeLabels[neighbor]) == 0:
            continue
        if len(nodeLabels[node]) == 0 and not shouldPropagate(0.5):
            continue
        for label in neighborLabels:
            if label not in labelRank:
                labelRank[label] = 0
            labelRank[label] = labelRank[label] + 1
 
    if len(labelRank) == 0:
        return nodeLabels

    maxKeys = findKeysWithMax(labelRank)
    nodeLabels[node] = maxKeys
    return nodeLabels

def generateColorMap(nodes, nodeLabels):
    colorMap = {}
    for node in nodes:
        colorMap[nodeLabels[node][0]] = "#%06x" % random.randint(0, 0xFFFFFF)
    colorMap["none"] = "#%06x" % random.randint(0, 0xFFFFFF)
    return colorMap
    
def findLabelOccupations(nodeLabels):
    labelOccupation = {}
    for n in nodeLabels:
        labels = nodeLabels[n]
        for l in labels:
            if l not in labelOccupation:
                labelOccupation[l] = 0
            labelOccupation[l] = labelOccupation[l]+1
    return labelOccupation

def isInactive(node, nodeLabels):
    if node not in nodeLabels or len(nodeLabels[node]) == 0:
        return True
    return False

def labelPropagation(ug, nodeLabels, colorMap):
    beforePropagation = {}
    afterPropagation = copy.deepcopy(nodeLabels)
    Nvt = {}

    nxG = networkXGraph(ug)
    spring_pos = nx.spring_layout(nxG, seed=2) 
    colors = "bgrcmykw"

    count = 0
    while beforePropagation != afterPropagation:
        count+=1
        beforePropagation = copy.deepcopy(afterPropagation)
        for node in ug:
            afterPropagation = extendedKroneckerDelta(ug, node, afterPropagation)
            # For each t update Nv
            labelOccupations = findLabelOccupations(afterPropagation)
            for l in labelOccupations:
                if l not in Nvt:
                    Nvt[l] = []
                Nvt[l].append(labelOccupations[l])
        
        # Drawing label propagation process
        plt.clf()
        plt.title('Iteration {}'.format(count))
        color_index = 0
        communities = extractCommnunity(afterPropagation)

        for label in communities:
            community = communities[label]
            noMember = len(community)
            displayLabel = "%s - %d" % (label, noMember)
            nx.draw_networkx_nodes(nxG, spring_pos, nodelist=community, node_color=colorMap[label], alpha=0.4, label=displayLabel)
            color_index += 1

        nx.draw_networkx_edges(nxG, spring_pos, style='dashed', width = 0.5)

        # Put a legend to the right of the current axis
        plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
        plt.pause(0.00000001)
        plt.draw()
            
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
    print("seed nodes", seedNodes)

    # Assign label to seed nodes
    seedNodeLabels = assignLabel(seedNodes)
    # print("seed node labels", seedNodeLabels)
    print("Number of seed nodes", len(seedNodeLabels))

    colorMap = generateColorMap(seedNodes, seedNodeLabels)

    # Non-seed nodes should be assigned with empty label set
    nodeLabels = copy.deepcopy(seedNodeLabels)
    nodes, deg = metadata(ug)
    for n in nodes:
        if n not in nodeLabels:
            nodeLabels[n] = []

    # Start propagating labels until labels distribution is stable
    propagatedLabels, Nvt = labelPropagation(ug, nodeLabels, colorMap)
    # print(propagatedLabels)

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

def networkXGraph(ug):
    G=nx.Graph()
    for start in ug:
        ends = ug[start]
        for end in ends:
            G.add_edge(start, end)
    return G

ug = readDataSet()
propagatedLabels, nodeCentralities, nodeLabels = imlpa(ug)
celebs = getNMostInfluenceNode(nodeCentralities, nodeLabels, 3)
# print(celebs)
print("Number of celebrities", len(celebs))

communities = extractCommnunity(propagatedLabels)
print("communities", communities)
print("Number of communities", len(communities))

plt.show()