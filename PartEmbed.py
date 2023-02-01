import networkx
import pymetis

# Given a networkx graph, embed the nodes of the graph
# where k is the number of partitions to use in embedding process
def partEmbed(G,k):
    embeds = []
    adjncy,xadj,vweights,eweights = getAdjLists(G)
    cutcount, part_vert = pymetis.part_graph(k,xadj=xadj,adjncy=adjncy,vweights=vweights,eweights=eweights)

    return embeds

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




if __name__ == "__main__":
    G = networkx.gnp_random_graph(10,0.2,seed=27)
    embeds = partEmbed(G,3)
    print(embeds)