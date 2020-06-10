import networkx as nx
#G = nx.karate_club_graph()
import matplotlib.pyplot as plt 

#nx.draw(G, with_label= True, node_color= 'lightblue', edge_color= 'gray')
#G.degree()[33] # grado del nodo 33
#G.degree(33) # Also work



#Erdős-Rényi graph
from scipy.stats import bernoulli
#bernoulli.rvs(p= 0.2)
N = 20
p = 0.2


def er_graph(N, p):
	''' Er graph'''
	G = nx.Graph()
	G.add_nodes_from(range(N))

	for node1 in G.nodes():
		for node2 in G.nodes():
			if node1 < node2 and bernoulli.rvs(p= p):
				G.add_edge(node1, node2)
	return G

def plot_degree_distribution(G):
	plt.hist([d for n, d in G.degree()], histtype="step")
	plt.xlabel("Degree $k$")
	plt.ylabel("$P(k)$")
	plt.title("Degree distribution")
	plt.show()

g1 = nx.erdos_renyi_graph(100, 0.03)

plot_degree_distribution(g1)





