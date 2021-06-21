from typing import final
import pandas as pd
import numpy as np
from collections import Counter
import networkx as nx
from data import Data
from cdlib import evaluation, algorithms, NodeClustering
from functools import reduce
import stanza
from nltk import word_tokenize, ngrams
from matplotlib import pyplot as plt
from rouge import Rouge
from zss import simple_distance, Node
import logging


class GNN:
    def __init__(self):
        self.data = []
        self.G = None
        self.G_ngrs = None
        self.pos_tokenized = []

    def set_data(self, dir):
        d = Data(dir)
        d.read_txt_data()
        d.build_model()
        self.data = d.stemmed_docs
        self.orig_data = d.data
        self.data_uncleaned = d.data_uncleaned
        self.data_ngrams = d.ngrams
        self.char_ngrams = d.char_ngrams
        self.data_tokenized = d.data_tokenized

    def doc_sim(self, doc1, doc2):
        len_d1 = len(doc1)
        len_d2 = len(doc2)
        sim_words_set = set(doc1).intersection(set(doc2))
        sim_words_len = len(list(sim_words_set))

        len_d1_log = np.log(len_d1) if len_d1 > 0 else 0
        len_d2_log = np.log(len_d2) if len_d2 > 0 else 0

        len_sum = len_d1_log+len_d2_log
        sim = sim_words_len/len_sum if len_sum > 0.0 and sim_words_len > 0 else 0

        return sim

    @staticmethod
    def get_ted_sim(doc1, doc2):
        return 1/(1+simple_distance(doc1, doc2))

    def build_graph(self, metric, threshold=0.00001):
        nodes = list(range(len(self.data)))
        G = nx.Graph()

        G.add_nodes_from(nodes)

        if metric == "ted":
            deps = self.build_dependency_trees()

        weights = []

        for i in range(0, len(self.data)):
            for j in range(i+1, len(self.data)):

                if metric == "overlap":
                    weights.append(((i, j), self.doc_sim(
                        self.data[i], self.data[j])))
                else:
                    weights.append(
                        ((i, j), self.get_ted_sim(deps[i], deps[j])))

        weights = list(sorted(
            weights, key=lambda x: x[1], reverse=True if metric == "overlap" else False))
        cutoff = int(len(weights) * threshold)

        for edge, weight in weights[:cutoff]:
            G.add_edge(*edge, weight=weight)

        self.G = G

    @staticmethod
    def log_graphs(graph):
        print(
            f"Graph has {graph.number_of_nodes()} nodes, {graph.number_of_nodes()} edges.")

    def build_graph_ngrams(self):

        G = nx.MultiGraph()
        G.add_nodes_from(list(self.char_ngrams[0]))

        for item in self.char_ngrams[1:]:
            G.add_node(item[1])

        G.add_edges_from(self.char_ngrams)

        print(G.nodes)
        self.G_ngrs = G
        self.log_graphs(self.G_ngrs)

    def build_word_graph(self, data):
        g = nx.DiGraph()
        g.add_nodes_from(["S", "E"])

        last_tokens = []
        edges_to_be_added = []

        for idx, sentence in enumerate(data):
            nodes = list(g.nodes)
            tokens = word_tokenize(sentence)

            queue = []

            for token in tokens:
                if token not in queue:
                    queue.append(token)

            g.add_nodes_from(queue)

            edges_to_be_added.append(("S", tokens[0]))
            edges_to_be_added.append((tokens[-1], 'E'))

            ngrams_list = list(ngrams(tokens, 2))
            edges_to_be_added.extend(ngrams_list)

            last_tokens.append(tokens[0])

            edges_to_be_added.append((last_tokens[idx-1], tokens[0]))

        unique, cnts = np.unique(edges_to_be_added, return_counts=True, axis=0)

        for idx, edge in enumerate(unique):
            g.add_edge(*edge, weight=(1/cnts[idx]))

        return g

    @staticmethod
    def sort_clique_items_by_closeness(closeness, cliques):
        cands = []

        for clique in list(filter(lambda x: len(x) > 1, cliques)):
            mapped = map(lambda x: (x, closeness[x]), clique)
            top = list(sorted(mapped, key=lambda x: x[1]))

            idx = 0
            top_cand = top[idx]

            while True:
                if top_cand not in cands:
                    cands.append(top_cand)
                    break
                else:

                    idx += 1
                    if idx < len(top):
                        top_cand = top[idx]
                    else:
                        break

        return {k: v for k, v in cands}

    def summarize(self, sorting_method="pagerank"):
        if sorting_method == "pagerank":
            try:
                scores = nx.pagerank(self.G, max_iter=1000)
            except:
                scores = {}
        elif sorting_method == "hits":
            try:
                scores = nx.hits(self.G, max_iter=1000)[0]
            except:
                scores = {}
        elif sorting_method == "closeness":
            scores = nx.closeness_centrality(self.G)
        elif sorting_method == "betweenness":
            scores = nx.betweenness_centrality(self.G)
        elif sorting_method == "degree":
            scores = nx.degree_centrality(self.G)
        elif sorting_method == "cliques":
            closeness = nx.closeness_centrality(self.G)
            cliques = list(nx.find_cliques(self.G))
            scores = self.sort_clique_items_by_closeness(closeness, cliques)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        final_doc = []

        for idx, _ in sorted_docs[:5]:

            final_doc.append(self.orig_data[idx])

        return ".".join(final_doc)

    def read_docs_by_idx(self, collection):
        final_doc = []

        for idx in collection:

            final_doc.append(self.orig_data[idx])

        return ". ".join(final_doc)

    @staticmethod
    def tree_to_graph(deps):
        g = nx.Graph()

        heads = []
        ids = []

        for dep in deps:
            heads.append(dep['head'])
            ids.append(dep['id'])

        g.add_nodes_from(ids)

        for idx, _ in enumerate(ids):
            if heads[idx] != 0:
                g.add_edge(ids[idx], heads[idx])

        return g

    @staticmethod
    def build_zss_tree(deps):
        labels = {}
        tree = {}
        root = None

        for dep in deps:
            if dep['head'] == 0:
                root = dep['id']
                labels[dep['id']] = dep['text']

            if True:
                if dep['head'] in tree:
                    tree[dep['head']].append(dep['id'])
                else:
                    tree[dep['head']] = [dep['id']]

                labels[dep['id']] = dep['text']

        def iterative(coll):
            out = []
            for child in coll:
                if child in tree:
                    out.append(Node(labels[child], iterative(tree[child])))
                else:
                    out.append(Node(labels[child], []))

            return out

        if len(deps) > 1:
            final_tree = Node(labels[root], iterative(tree[root]))
        else:
            final_tree = Node(labels[1], [])

        return final_tree

    def build_dependency_trees(self):
        deps = []
        term_freqs = []
        nlp = stanza.Pipeline(
            processors="tokenize, pos, lemma, depparse", tokenize_pretokenized=True, verbose=False)

        doc = nlp(self.data_tokenized)

        for idx, sent in enumerate(doc.to_dict()):
            deps.append(self.build_zss_tree(sent))

        return deps

    def evaluate(self, methods, true_summary):
        results = []

        for method in methods:
            summary = gnn.summarize(sorting_method=method)
            summary = summary if summary != "" else " "
            results.append(Rouge().get_scores(summary, true_summary))

        return results


batch_size = 30

methods = ["pagerank", "hits", "closeness",
           "betweenness", "degree", "cliques"]
results = [[] for _ in range(len(methods))]

logging.basicConfig(filename='results2.log',
                    level=logging.DEBUG, format='%(asctime)s %(message)s')

batches = 100

avg_results_f = [[[], [], []] for _ in range(len(methods))]
avg_results_r = [[[], [], []] for _ in range(len(methods))]

# for testing all at once and evaluating all at once

for i in range(3001):
    print(i)
    gnn = GNN()
    gnn.set_data(f'cnn_daily/{i}_article.txt')
    gnn.build_graph(metric="overlap", threshold=0.1)

    f = open(f"cnn_daily/{i}_summ.txt").readlines()
    result = gnn.evaluate(methods, f[0])
    logging.info(result)

    for idx in range(len(methods)):
        results[idx].append(result[idx])

for idx in range(6):
    r1, r2, rl = [], [], []
    f1, f2, fl = [], [], []

    for item in results[idx]:
        r1_score = item[0]['rouge-1']['r']
        r2_score = item[0]['rouge-2']['r']
        rl_score = item[0]['rouge-l']['r']

        if r1 != 0.0:
            r1.append(r1_score)

        if r2 != 0.0:
            r2.append(r2_score)

        if rl != 0.0:
            rl.append(rl_score)

        f1_score = item[0]['rouge-1']['f']
        f2_score = item[0]['rouge-2']['f']
        fl_score = item[0]['rouge-l']['f']

        if f1 != 0.0:
            f1.append(f1_score)

        if f2 != 0.0:
            f2.append(f2_score)

        if fl != 0.0:
            fl.append(fl_score)

    r1 = sum(r1)/len(r1)
    r2 = sum(r2)/len(r2)
    rl = sum(rl)/len(rl)

    f1 = sum(f1)/len(f1)
    f2 = sum(f2)/len(f2)
    fl = sum(fl)/len(fl)

    print(methods[idx], f"{r1} / {r2} / {rl} \n", f"{f1} / {f2} / {fl} \n\n")


# for testing in different batches

for size in range(1, batches + 1):

    for i in range((size-1)*batch_size, ((size-1)*batch_size)+batch_size):
        print(i)
        gnn = GNN()
        gnn.set_data(f'cnn_daily/{i}_article.txt')
        gnn.build_graph(metric="overlap", threshold=0.1)

        f = open(f"cnn_daily/{i}_summ.txt").readlines()
        result = gnn.evaluate(methods, f[0])
        logging.info(result)

        for idx in range(len(methods)):
            results[idx].append(result[idx])

    for idx in range(len(methods)):
        r1, r2, rl = [], [], []
        f1, f2, fl = [], [], []

        for item in results[idx]:
            r1_score = item[0]['rouge-1']['r']
            r2_score = item[0]['rouge-2']['r']
            rl_score = item[0]['rouge-l']['r']

            if r1 != 0.0:
                r1.append(r1_score)

            if r2 != 0.0:
                r2.append(r2_score)

            if rl != 0.0:
                rl.append(rl_score)

            f1_score = item[0]['rouge-1']['f']
            f2_score = item[0]['rouge-2']['f']
            fl_score = item[0]['rouge-l']['f']

            if f1 != 0.0:
                f1.append(f1_score)

            if f2 != 0.0:
                f2.append(f2_score)

            if fl != 0.0:
                fl.append(fl_score)

        r1 = sum(r1)/len(r1)
        r2 = sum(r2)/len(r2)
        rl = sum(rl)/len(rl)

        f1 = sum(f1)/len(f1)
        f2 = sum(f2)/len(f2)
        fl = sum(fl)/len(fl)

        avg_results_r[idx][0].append(r1)
        avg_results_r[idx][1].append(r2)
        avg_results_r[idx][2].append(rl)

        avg_results_f[idx][0].append(f1)
        avg_results_f[idx][1].append(f2)
        avg_results_f[idx][2].append(fl)

for idx in range(len(methods)):
    print("f for", methods[idx], sum(avg_results_f[idx][0])/len(avg_results_f[idx][0]), sum(avg_results_f[idx]
                                                                                            [1])/len(avg_results_f[idx][1]), sum(avg_results_f[idx][1])/len(avg_results_r[idx][1]))
    print("r for", methods[idx], sum(avg_results_r[idx][0])/len(avg_results_r[idx][0]), sum(avg_results_r[idx]
                                                                                            [1])/len(avg_results_r[idx][1]), sum(avg_results_r[idx][2])/len(avg_results_r[idx][2]))
