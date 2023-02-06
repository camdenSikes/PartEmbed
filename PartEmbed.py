import networkx
import pymetis
import math
import numpy as np
import node2vec.src.node2vec as node2vec
from gensim.models import Word2Vec
from scipy.io import loadmat
import scipy.sparse

# Given a networkx graph, embed the nodes of the graph
def partEmbed(G):
    #TODO: how do we decide this value?
    delta = 1*(1/len(G.nodes))
    k = math.ceil(math.sqrt(len(G.nodes)))
    adjncy,xadj,vweights,eweights = getAdjLists(G)
    cutcount, part_vert = pymetis.part_graph(k,xadj=xadj,adjncy=adjncy,vweights=vweights,eweights=eweights)
    #TODO: cutcount is much higher than it should be
    #assert cutcount == k
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
            newVecs[i] = 0.5*(G.nodes[i]['embed'] + neighborSum/numNeighbors)
            delta += np.linalg.norm(G.nodes[i]['embed'] - newVecs[i])
        for i in G.nodes():
            G.nodes[i]['embed'] = newVecs[i]
        delta = delta/len(G.nodes())
    return


#compute the node2vec embeddings of a graph, using ref implementation
def computeNode2Vec(AG):
    #TODO: deal with hyperparameters, currently just doing the default
    returnHyperparam = 1
    inoutHyperparam = 1
    num_walks = 10
    walk_length = 80
    window_size = 10
    dimensions = 128
    iter = 1
    workers = 8

    G = node2vec.Graph(AG, False, returnHyperparam, inoutHyperparam)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    #Had to add list because of python 2/3 shenanigans
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, epochs=iter)
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





if __name__ == "__main__":
    G = networkx.gnp_random_graph(10,0.2,seed=27)

    mat_variables = loadmat("citeseer.mat")
    mat_matrix = mat_variables["network"]
    G = networkx.Graph(mat_matrix)

    G = partEmbed(G)
    print(G)
    saveEmbeddingMatrix(G,"partEmbedCiteseer.npy")