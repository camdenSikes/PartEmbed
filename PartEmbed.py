import networkx
from pygsp import graphs as pygraphs
import pymetis
import math
import numpy as np
import node2vec
from gensim.models import Word2Vec
from scipy.io import loadmat
import scipy.sparse
import g_coarsening

# Given a networkx graph, embed the nodes of the graph
def partEmbed(G):
    #TODO: how do we decide this value?
    delta = math.sqrt(len(G.nodes))*(1/len(G.nodes))
    k = math.ceil(math.sqrt(len(G.nodes)))
    adjncy,xadj,vweights,eweights = getAdjLists(G)
    cutcount, part_vert = pymetis.part_graph(k,xadj=xadj,adjncy=adjncy)
    #generate abstract graph
    A = scipy.sparse.lil_array((k,k),dtype=np.int32)
    for i, nbrsdict in G.adjacency():
        for j, eattr in nbrsdict.items():
            A[part_vert[i],part_vert[j]] += 1
    AG = networkx.Graph(A)
    #run node2vec on abstract graph
    # note: the alias method mentioned in the paper is already implemented here
    nodevecs = computeNode2Vec(AG)
    #Add embeddings from abstract graph to original
    for node in G.nodes:
        G.nodes[node]['embed'] = nodevecs[part_vert[node]]
    #embedding propagation
    propagate(G, delta)
    return G

#propagate the embeddings
def propagate(G,deltaLimit):
    #TODO: is it slower to store/access graph properties? could just use list
    delta = deltaLimit+1
    shape = np.shape(G.nodes[0]['embed'])
    newVecs = list(G.nodes())
    while (delta>deltaLimit):
        delta = 0
        for i, nbrsdict in G.adjacency():
            neighborSum = np.zeros(shape)
            numNeighbors = 0
            for j, eattr in nbrsdict.items():
                neighborSum += G.nodes[j]['embed']
                numNeighbors += 1
            if(numNeighbors == 0):
                newVecs[i] = G.nodes[i]['embed']
            else:
                newVecs[i] = 0.5*(G.nodes[i]['embed'] + neighborSum/numNeighbors)
            delta += np.linalg.norm(G.nodes[i]['embed'] - newVecs[i])
        for i in G.nodes():
            G.nodes[i]['embed'] = newVecs[i]
        delta = delta/len(G.nodes())
    return

def propagateIter(G):
    shape = np.shape(G.nodes[0]['embed'])
    newVecs = list(G.nodes())
    for i, nbrsdict in G.adjacency():
        neighborSum = np.zeros(shape)
        numNeighbors = 0
        for j, eattr in nbrsdict.items():
            neighborSum += G.nodes[j]['embed']
            numNeighbors += 1
        if(numNeighbors == 0):
            newVecs[i] = G.nodes[i]['embed']
        else:
            newVecs[i] = 0.5*(G.nodes[i]['embed'] + neighborSum/numNeighbors)
    for i in G.nodes():
        G.nodes[i]['embed'] = newVecs[i]


#compute the node2vec embeddings of a graph
def computeNode2Vec(AG):
    #TODO: deal with hyperparameters, currently just doing the default
    p = 0.25
    q = 0.25
    num_walks = 16
    walk_length = 100
    window_size = 16
    dimensions = 128
    iter = 1
    workers = 8

    n2v = node2vec.Node2Vec(AG, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers,p = p, q = q)
    model = n2v.fit(window = window_size, min_count=1, batch_words=4, epochs = iter)
    return model.wv


#Given a networkx graph, gets adjlists in format for pymetis
def getAdjLists(G):
    adjncy = []
    xadj = []
    vweights = []
    eweights = []
    curInd = 0
    for i, nbrsdict in G.adjacency():
        xadj.append(curInd)
        #vweights.append(G.nodes[i]['weight'])
        for j, eattr in nbrsdict.items():
            adjncy.append(j)
            if "weight" in eattr:
                eweights.append(eattr['weight'])
            curInd += 1
    xadj.append(curInd)

    return (adjncy,xadj,vweights,eweights)

#Given embedded graph, save a numpy matrix of the embeddings
def saveEmbeddingMatrix(G,filename):
    embeddings = np.ndarray(shape=(len(G.nodes), len(G.nodes[0]['embed'])), dtype=np.float32)
    for i in G.nodes:
        embeddings[i] = G.nodes[i]['embed']
    np.save(filename,embeddings)


def coarsenEmbed(G):
    pyG = pygraphs.Graph(networkx.adjacency_matrix(G))
    C,Gc,Call,Gall = g_coarsening.coarsen(pyG,r=0.9,method='heavy_edge',max_levels=50)
    Gcnx = networkx.Graph(Gc.W.todense())
    nodevecs = computeNode2Vec(Gcnx)
    embedarray = np.zeros((np.shape(C)[0],len(nodevecs[0])))
    for i in range(len(embedarray)):
        embedarray[i] = nodevecs[i]
    bigembedarray = np.transpose(C) * embedarray
    #Add embeddings from abstract graph to original
    for node in G.nodes:
        G.nodes[node]['embed'] = bigembedarray[node]
    return G

def coarsenEmbedRec(G):
    pyG = pygraphs.Graph(networkx.adjacency_matrix(G))
    C,Gc,Call,Gall = g_coarsening.coarsen(pyG,r=0.9,method='heavy_edge',max_levels=2)
    Gcnx = networkx.Graph(Gc.W.todense())
    nodevecs = computeNode2Vec(Gcnx)
    embedarray = np.zeros((np.shape(C)[0],len(nodevecs[0])))
    for i in range(len(embedarray)):
        embedarray[i] = nodevecs[i]

    for i in range(len(Call)):
        ind = len(Call) - 1 - i
        embedarray = np.transpose(Call[ind]) * embedarray
        #TODO: This is inefficient, can improve by not swapping each time
        tempGraph = networkx.Graph(Gall[ind].W.todense())
        for node in tempGraph.nodes:
            tempGraph.nodes[node]['embed'] = embedarray[node]
        #embedding propagation
        #TODO: how do we decide this value?
        delta = math.sqrt(len(tempGraph.nodes))*(1/len(tempGraph.nodes))
        #propagateIter(tempGraph)
        propagate(tempGraph,delta)
        for node in tempGraph.nodes:
            embedarray[node] = tempGraph.nodes[node]['embed']
    return tempGraph






if __name__ == "__main__":
    G = networkx.gnp_random_graph(100,0.2,seed=27)

    mat_variables = loadmat("citeseer.mat")
    mat_matrix = mat_variables["network"]
    G = networkx.Graph(mat_matrix)

    #If I ever need to compute pure node2vec
    #nodevecs = computeNode2Vec(G)
    #embeddings = np.ndarray(shape=(len(G.nodes), len(nodevecs[0])), dtype=np.float32)
    #for i in G.nodes:
    #    embeddings[i] = nodevecs[i]
    #np.save("node2vecCiteseer.npy",embeddings)

    #G = partEmbed(G)
    G = coarsenEmbedRec(G)
    saveEmbeddingMatrix(G,"partEmbedCiteseer.npy")