import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random
import numpy as np

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

def create_innovative_agents(n_agents, p_innovative):
    return np.random.choice([0,1], size=n_agents, p=[1-p_innovative,p_innovative])

def experiment_friend_of_friend(n_iterations=1):
    agent_types = create_innovative_agents(n_agents=100,p_innovative=0.1)
    agents = range(len(agent_types))
    g = nx.Graph()
    for a in agents:
        g.add_node(a, agent_type=agent_types[a])
    print(g.nodes(data=True))
    stranger_connect_prob = 0.2
    conservative_friend_of_friend_connect_prob = 0.5
    innovative_friend_of_friend_connect_prob = 0.2
    pairs = list(combinations(agents,2))
    # TODO: Shuffle combinations list?
    for it in range(n_iterations):
        # print(f"Iteration: {it}")
        for i,j in pairs:
            # print(f"Evaluating edge {i,j}")
            if g.has_edge(i,j):
                # print(f"- Edge {i,j} already exists. Only possible when running multiple iterations.")
                continue
            # Check if friend of a friend
            common_neighbors = list(nx.common_neighbors(g,i,j))
            connect_prob = None
            if not common_neighbors:
                # print(f"- No common neighbors: {common_neighbors}")
                connect_prob = stranger_connect_prob
            else:
                # If there is a neighbor in common
                # print(f"- Common neighbors: {common_neighbors}")
                # Use innovative prob if one of the nodes is innovative
                if g.nodes[i]["agent_type"]==1 or g.nodes[j]["agent_type"]==1:
                    # print(f"- One of the nodes {i,j} is innovative")
                    connect_prob = innovative_friend_of_friend_connect_prob
                else:
                    # print(f"- None of the nodes {i,j} is innovative")
                    connect_prob = conservative_friend_of_friend_connect_prob
            if random.random() < connect_prob:
                # print(f"- Add edge {i,j}")
                g.add_edge(i,j)
    print(nx.clustering(g))
    # TODO: Computer clusterings per agent type


    

#experiment_small_world()
#experiment_connected_cliques()
experiment_friend_of_friend(2)