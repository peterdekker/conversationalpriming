import networkx as nx
import matplotlib.pyplot as plt

N_NODES = 100
N_NEIGHBOURS = 4

def check_small_world(graph):
    s = nx.sigma(graph)
    o = nx.omega(graph)
    avg_sp = nx.average_shortest_path_length(graph)
    print(f"Small world sigma: {s}, omega: {o}, avg shortest path: {avg_sp}")


    check_clustering_degree(graph)
    print("")

def check_clustering_degree(graph):
    print(f"Degree: {nx.degree_histogram(graph)}")
    print(f"avg clustering: {nx.average_clustering(graph)}.")
    print(f"Clustering coefficients per node: {nx.clustering(graph)}")
    print(f"Betweenness centrality: {nx.betweenness_centrality(graph)}")

def draw(graph):
    nx.draw(graph)
    plt.show()
    plt.clf()

def experiment_small_world():
    print("Small world")
    small_world = nx.connected_watts_strogatz_graph(n=N_NODES, k=N_NEIGHBOURS, p=0.5, tries=20)
    check_small_world(small_world)
    draw(small_world)

    print("Random")
    random = nx.random_reference(small_world)
    check_small_world(random)
    draw(random)

    print("Lattice reference")
    lattice = nx.random_reference(small_world)
    check_small_world(lattice)
    draw(lattice)

    print("Scale-free [directed]")
    scale_free = nx.scale_free_graph(n=N_NODES)
    print(nx.degree_histogram(scale_free))
    draw(scale_free)

# def create_connected_cliques_graph():
#     half_nodes = N_NODES // 2
#     clique1 = nx.complete_graph(half_nodes)
#     connecting_node1 = list(clique1.nodes())[0]
#     clique2 = nx.relabel_nodes(clique1, mapping=lambda x:x+half_nodes)
#     connecting_node2 = list(clique2.nodes())[0]
#     combined = nx.compose(clique1,clique2)
#     combined.add_edge(connecting_node1,connecting_node2)
#     return combined

def experiment_barbell():
    # connected_cliques = create_connected_cliques_graph()
    # check_clustering_degree(connected_cliques)
    barbell = nx.barbell_graph(50,0)
    check_clustering_degree(barbell)
    

#experiment_small_world()
experiment_connected_cliques()