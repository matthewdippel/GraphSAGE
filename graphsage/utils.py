from __future__ import print_function

import numpy as np
import random
import json
import sys
import os
from collections import defaultdict

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=5
N_WALKS=50

class FakeGraph:
    """
    provide a similar interface to networkx but avoid the overhead. will this work?
    """
    def __init__(self):
        self.node = {}
        self._edges = defaultdict(lambda: defaultdict(lambda: {}))

    def nodes(self):
        return self.node.keys()

    def edges(self):
        for i in self._edges:
            for j in self._edges[i]:
                yield (i, j)

    def add_edge(self, i, j):
        self._edges[i][j] = {}

    def add_node(self, node, data):
        self.node[node] = data

    def __getitem__(self, key):
        return self._edges[key]

    def neighbors(self, node):
        for j in self._edges[node]:
            yield j
    



def load_data(prefix, normalize=True, load_walks=False):
    # Matt: I think the OOM problems are coming from this representation of the data
    # So you'll need to make a different data format, and write the code to read it into the FakeGraph. 
    # Below is just an example conversion of the networkx graph into the wrapper
    # To show that the training code still works
    G_data = json.load(open(prefix + "-G.json"))
    G_pre = json_graph.node_link_graph(G_data)
    G = FakeGraph()
    for edge in G_pre.edges():
        G.add_edge(edge[0], edge[1])

    for node in G_pre.nodes():
        G.add_node(node, G_pre.node[node])

    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    #broken_count = 0
    #for node in G.nodes():
        #if not 'val' in G.node[node] or not 'test' in G.node[node]:
            #G.remove_node(node)
            #broken_count += 1
    #print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map

def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
