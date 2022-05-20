import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random
import numpy as np
import pandas as pd

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
    small_world = nx.connected_watts_strogatz_graph(
        n=N_NODES, k=N_NEIGHBOURS, p=0.5, tries=20)
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
    barbell = nx.barbell_graph(50, 0)
    check_clustering_degree(barbell)


def create_innovative_agents(n_agents, p_innovating):
    agent_types = np.random.choice([False, True], size=n_agents, p=[
                                   1-p_innovating, p_innovating])
    agents = range(len(agent_types))
    return agent_types, agents


def experiment_friend_of_friend(n_agents, p_innovating, stranger_connect_prob, conservating_friend_of_friend_connect_prob,
                                innovating_friend_of_friend_connect_prob, n_iterations=1, dodraw=True):
    agent_types, agents = create_innovative_agents(n_agents, p_innovating)
    g = create_network_friend_of_friend(stranger_connect_prob, conservating_friend_of_friend_connect_prob,
                                        innovating_friend_of_friend_connect_prob, n_iterations, agent_types, agents)
    clustering_coeffs = nx.clustering(g)
    ids_innovating = [id for id, attrs in g.nodes(
        data=True) if attrs["innovating"] == True]
    ids_conservating = [id for id, attrs in g.nodes(
        data=True) if attrs["innovating"] == False]
    clustering_innovating_mean = np.mean(
        [clustering_coeffs[id] for id in ids_innovating])
    clustering_conservating_mean = np.mean(
        [clustering_coeffs[id] for id in ids_conservating])
    print(
        f"Mean clustering coefficient, innovating: {clustering_innovating_mean}, conservating: {clustering_conservating_mean}")

    degree_innovating_mean = np.mean([g.degree[id] for id in ids_innovating])
    degree_conservating_mean = np.mean(
        [g.degree[id] for id in ids_conservating])
    print(
        f"Degree, innovating: {degree_innovating_mean}, conservating: {degree_conservating_mean}")
    if dodraw:
        draw(g)


def experiment_friend_of_friend_fixed_degree(n_agents, p_innovating, stranger_connect_prob, conservating_friend_of_friend_connect_prob,
                                             innovating_friend_of_friend_connect_prob, max_degree, dodraw=True):
    agent_types, agents = create_innovative_agents(n_agents, p_innovating)
    g = create_network_friend_of_friend_fixed_degree(
        stranger_connect_prob, conservating_friend_of_friend_connect_prob, innovating_friend_of_friend_connect_prob, max_degree, agent_types, agents)
    clustering_coeffs = nx.clustering(g)
    ids_innovating = [id for id, attrs in g.nodes(
        data=True) if attrs["innovating"] == True]
    ids_conservating = [id for id, attrs in g.nodes(
        data=True) if attrs["innovating"] == False]
    clustering_innovating_mean = np.mean(
        [clustering_coeffs[id] for id in ids_innovating])
    clustering_conservating_mean = np.mean(
        [clustering_coeffs[id] for id in ids_conservating])
    # print(
    #     f"Mean clustering coefficient, innovating: {clustering_innovating_mean}, conservating: {clustering_conservating_mean}")

    degree_innovating_mean = np.mean([g.degree[id] for id in ids_innovating])
    degree_conservating_mean = np.mean(
        [g.degree[id] for id in ids_conservating])
    # print(
    #     f"Degree, innovating: {degree_innovating_mean}, conservating: {degree_conservating_mean}")
    if dodraw:
        #draw(g)
        pos = nx.kamada_kawai_layout(g)
        nodes=g.nodes()
        colors = ids_innovating = ["green" if attrs["innovating"] == True else "grey" for id, attrs in g.nodes(data=True)]
        ec = nx.draw_networkx_edges(g, pos, alpha=0.2)
        nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors, 
                                    node_size=100)
        plt.savefig("friendoffriend.png",dpi=300)
    record =  {"n_agents": n_agents, "stranger_connect_prob": stranger_connect_prob, "conservating_friend_of_friend_connect_prob": conservating_friend_of_friend_connect_prob,
            "innovating_friend_of_friend_connect_prob": innovating_friend_of_friend_connect_prob, "max_degree": max_degree,
            "cl_inn": clustering_innovating_mean, "cl_con": clustering_conservating_mean, "deg_inn": degree_innovating_mean, "deg_conn": degree_conservating_mean}
    print(record)
    return record


def explore_params_fof_fixed():
    records = []
    for n_agents in [100, 200]:
        for max_degree in [5,10]:
            for stranger_connect_prob in np.linspace(0.1, 1, 10):
                for innovating_friend_of_friend_connect_prob in np.linspace(0.1, 1, 10):
                    for conservating_friend_of_friend_connect_prob in np.linspace(0.1, 1, 10):
                        if stranger_connect_prob <= innovating_friend_of_friend_connect_prob and innovating_friend_of_friend_connect_prob <= conservating_friend_of_friend_connect_prob:
                            for i in range(10):
                                result = experiment_friend_of_friend_fixed_degree(n_agents=n_agents, p_innovating=0.2, stranger_connect_prob=stranger_connect_prob, conservating_friend_of_friend_connect_prob=conservating_friend_of_friend_connect_prob,
                                                                        innovating_friend_of_friend_connect_prob=innovating_friend_of_friend_connect_prob, max_degree=max_degree, dodraw=False)
                                records.append(result)
    df = pd.DataFrame.from_records(records)
    # Mean over 10 runs
    dfm = df.groupby(["n_agents", "stranger_connect_prob", "conservating_friend_of_friend_connect_prob", "innovating_friend_of_friend_connect_prob","max_degree"]).mean()
    dfm["cl_diff"] = dfm["cl_con"] - dfm["cl_inn"]
    for _, g in dfm.sort_values("cl_diff",ascending=False).groupby("n_agents"):
        print(g)
                        
                        


def create_network_friend_of_friend(stranger_connect_prob, conservating_friend_of_friend_connect_prob, innovating_friend_of_friend_connect_prob, n_iterations, agent_types, agents):
    g = nx.Graph()
    for a in agents:
        g.add_node(a, innovating=agent_types[a])
    pairs = list(combinations(agents, 2))
    # TODO: Make sure innovating and conservating agents have same degree
    # TODO: Shuffle combinations list?
    for it in range(n_iterations):
        # print(f"Iteration: {it}")
        for i, j in pairs:
            # print(f"Evaluating edge {i,j}")
            if g.has_edge(i, j):
                # print(f"- Edge {i,j} already exists. Only possible when running multiple iterations.")
                continue
            # Check if friend of a friend
            common_neighbors = list(nx.common_neighbors(g, i, j))
            connect_prob = None
            if not common_neighbors:
                # print(f"- No common neighbors: {common_neighbors}")
                connect_prob = stranger_connect_prob
            else:
                # If there is a neighbor in common
                # print(f"- Common neighbors: {common_neighbors}")
                # Use innovative prob if one of the nodes is innovative
                if g.nodes[i]["innovating"] == 1 or g.nodes[j]["innovating"] == 1:
                    # print(f"- One of the nodes {i,j} is innovative")
                    connect_prob = innovating_friend_of_friend_connect_prob
                else:
                    # print(f"- None of the nodes {i,j} is innovative")
                    connect_prob = conservating_friend_of_friend_connect_prob
            if random.random() < connect_prob:
                # print(f"- Add edge {i,j}")
                g.add_edge(i, j)
    return g


def get_agents_not_max_degree(g, max_degree):
    agents_not_max = [n for n, degree in g.degree if degree < max_degree]
    return agents_not_max


def create_network_friend_of_friend_fixed_degree(stranger_connect_prob, conservating_friend_of_friend_connect_prob, innovating_friend_of_friend_connect_prob, max_degree, agent_types, agents):
    max_retries = 100
    g = nx.Graph()
    for a in agents:
        g.add_node(a, innovating=agent_types[a])
    agents_not_full = get_agents_not_max_degree(g, max_degree)
    n_agents_not_full_iterations = []
    n_agents_not_full_prev = -1
    counter = 0
    while (agents_not_full):
        n_agents_not_full = len(agents_not_full)
        # If 100 iterations same number of not full agents, quit
        if n_agents_not_full == n_agents_not_full_prev:
            counter += 1
            if counter == max_retries:
                break
        else:
            counter = 0
        n_agents_not_full_prev = n_agents_not_full

        # Run only through not-full (not max degree) agents to save run time, this is no guarantee, agents can be filled during loop
        for i in agents_not_full:
            for j in agents_not_full:
                if i == j:
                    continue
                if g.has_edge(i, j):
                    continue
                if g.degree[i] == max_degree or g.degree[j] == max_degree:
                    continue
                # Check if friend of a friend
                common_neighbors = list(nx.common_neighbors(g, i, j))
                connect_prob = None
                if not common_neighbors:
                    # print(f"- No common neighbors: {common_neighbors}")
                    connect_prob = stranger_connect_prob
                else:
                    # If there is a neighbor in common
                    # print(f"- Common neighbors: {common_neighbors}")
                    # Use innovative prob if one of the nodes is innovative
                    if g.nodes[i]["innovating"] == 1 or g.nodes[j]["innovating"] == 1:
                        # print(f"- One of the nodes {i,j} is innovative")
                        connect_prob = innovating_friend_of_friend_connect_prob
                    else:
                        # print(f"- None of the nodes {i,j} is innovative")
                        connect_prob = conservating_friend_of_friend_connect_prob
                if random.random() < connect_prob:
                    # print(f"- Add edge {i,j}")
                    g.add_edge(i, j)
        agents_not_full = get_agents_not_max_degree(g, max_degree)

        # TODO: Recompute condition
    return g


def create_network_complete(n_agents, agent_types):
    graph = nx.complete_graph(n_agents)
    agent_types_dict = dict(enumerate(agent_types))
    nx.set_node_attributes(graph, agent_types_dict, name="innovating")
    return graph


# experiment_small_world()
# experiment_connected_cliques()
if __name__ == "__main__":
    #explore_params_fof_fixed()

    print("Fixed:")
    experiment_friend_of_friend_fixed_degree(n_agents=100, p_innovating=0.2, stranger_connect_prob=0.3, conservating_friend_of_friend_connect_prob=1.0,
                                             innovating_friend_of_friend_connect_prob=0.3, max_degree=10)
    # print("Non-fixed:")
    # experiment_friend_of_friend(n_agents=100, p_innovating=0.2, stranger_connect_prob=0.1, conservating_friend_of_friend_connect_prob=0.5,
    #                                 innovating_friend_of_friend_connect_prob=0.2, n_iterations=1)
