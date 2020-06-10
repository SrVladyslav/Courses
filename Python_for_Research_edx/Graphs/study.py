import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 

def plot_degree_distribution(G):
	plt.hist([d for n, d in G.degree()], histtype="step")
	plt.xlabel("Degree $k$")
	plt.ylabel("$P(k)$")
	plt.title("Degree distribution")
	#plt.show()

A1 = np.loadtxt("adj_allVillageRelationships_vilno_1.csv", delimiter=",")
A2 = np.loadtxt("adj_allVillageRelationships_vilno_2.csv", delimiter=",")

G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

def basic_net_stats(G):
	print("Number of nodes:", G.number_of_nodes())
	print("Number of edges: ",G.number_of_edges())
	print("Average degree:",np.mean([d for n, d in G.degree()]))

#basic_net_stats(G1)
#basic_net_stats(G2)

#plot_degree_distribution(G1)
#plot_degree_distribution(G2)
#plt.show()

# ==================== COnnected components==================

# Generator for out g1 components
gen = nx.connected_component_subgraphs(G1)
gen2 = nx.connected_component_subgraphs(G2)
g = gen.__next__()

G_LCC = max(nx.connected_component_subgraphs(G1), key=len)
G_LCC2 = max(nx.connected_component_subgraphs(G2), key=len)
G_LCC.number_of_nodes()

type(g)





# Printing all that 
# 
plf.figure()
nx.draw(G1_LCC, node_color = 'red', edge_color= 'gray', node_size= 20)
pls.show()