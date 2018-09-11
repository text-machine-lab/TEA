import copy
import numpy as np
import sys

DIRECTED = ('BEFORE', 'IBEFORE', 'AFTER', 'IAFTER', 'IS_INCLUDED', 'INCLUDES', 'BEGINS', 'BEGUN_BY', 'ENDS', 'ENDED_BY')

def cycle_exists(G):  # - G is a directed graph
    color = {u: "white" for u in G}  # - All nodes are initially white
    found_cycle = [False]  # - Define found_cycle as a list so we can change
    # its value per reference, see:
    # http://stackoverflow.com/questions/11222440/python-variable-reference-assignment
    for u in G:  # - Visit all nodes.
        if color[u] == "white":
            dfs_visit(G, u, color, found_cycle)
        if found_cycle[0]:
            break
    return found_cycle[0]


def dfs_visit(G, u, color, found_cycle):
    if found_cycle[0]:  # - Stop dfs if cycle is found.
        return
    color[u] = "gray"  # - Gray nodes are in the current path
    for v in G[u]:  # - Check neighbors, where G[u] is the adjacency list of u.
        if color[v] == "gray":  # - Case where a loop in the current path is present.
            found_cycle[0] = True
            # print "found cycle here:", u, v
            return
        if color[v] == "white":  # - Call dfs_visit recursively.
            dfs_visit(G, v, color, found_cycle)
    color[u] = "black"  # - Mark nod


def simplify_graph(G):
    """remove the scores, so the cycle_exits() function can work"""
    graph = copy.deepcopy(G)
    simplified = dict((k, graph[k][0]) for k in graph)

    # add dummy edges,so the cycle_exists() function works
    for source in simplified.keys():
        for target in simplified[source]:
            if target not in simplified:
                simplified[target] = []
    return simplified


def get_parents(G, u):
    """Given a graph and a node, return the parents nodes"""
    parents = []
    scores = []
    for v in G:
        if u in G[v][0]:
            parents.append(v)
            index = G[v][0].index(u)
            try:
                scores.append(G[v][1][index])
            except IndexError:
                print(v, index, G[v])
                sys.exit("IndexError: line 59")
    return parents, scores

def build_graph(id_pairs, labels, scores):
    # id_pairs = []
    # labels = []
    # scores = []
    # for id_pair in id_pair_index:
    #     id_pairs.append(id_pair)
    #     index = id_pair_index[id_pair]
    #     labels.append(input_labels[index])
    #     scores.append(input_scores[index])

    before_graph = {}  # {eid: [[eid_1, eid_2, ...],[score_1, score_2,...]}
    included_graph = {}
    begins_graph = {}
    ends_graph = {}

    for pair, label, score in zip(id_pairs, labels, scores):
        if label in ('BEFORE', 'IBEFORE'): # use bidrectional edges for 'SIMULTANEOUS'
            if pair[0] in before_graph:
                before_graph[pair[0]][0].append(pair[1])
                before_graph[pair[0]][1].append(score)
            else:
                before_graph[pair[0]] = [[pair[1]], [score]]
        if label in ('AFTER', 'IAFTER'):
            if pair[1] in before_graph:
                before_graph[pair[1]][0].append(pair[0])
                before_graph[pair[1]][1].append(score)
            else:
                before_graph[pair[1]] = [[pair[0]], [score]]
        if label == 'IS_INCLUDED':
            if pair[0] in included_graph:
                included_graph[pair[0]][0].append(pair[1])
                included_graph[pair[0]][1].append(score)
            else:
                included_graph[pair[0]] = [[pair[1]], [score]]
        if label == 'INCLUDES':
            if pair[1] in included_graph:
                included_graph[pair[1]][0].append(pair[0])
                included_graph[pair[1]][1].append(score)
            else:
                included_graph[pair[1]] = [[pair[0]], [score]]
        if label == 'BEGINS':
            if pair[0] in begins_graph:
                begins_graph[pair[0]][0].append(pair[1])
                begins_graph[pair[0]][1].append(score)
            else:
                begins_graph[pair[0]] = [[pair[1]], [score]]
        if label == 'BEGUN_BY':
            if pair[1] in begins_graph:
                begins_graph[pair[1]][0].append(pair[0])
                begins_graph[pair[1]][1].append(score)
            else:
                begins_graph[pair[1]] = [[pair[0]], [score]]
        if label == 'ENDS':
            if pair[0] in ends_graph:
                ends_graph[pair[0]][0].append(pair[1])
                ends_graph[pair[0]][1].append(score)
            else:
                ends_graph[pair[0]] = [[pair[1]], [score]]
        if label == 'ENDED_BY':
            if pair[1] in ends_graph:
                ends_graph[pair[1]][0].append(pair[0])
                ends_graph[pair[1]][1].append(score)
            else:
                ends_graph[pair[1]] = [[pair[0]], [score]]

    before_graph = integrate_simul(id_pairs, labels, scores, before_graph)
    included_graph = integrate_simul(id_pairs, labels, scores, included_graph)
    begins_graph = integrate_simul(id_pairs, labels, scores, begins_graph)
    ends_graph = integrate_simul(id_pairs, labels, scores, ends_graph)

    return before_graph, included_graph, begins_graph, ends_graph


def integrate_simul(id_pairs, labels, scores, graph):
    for pair, label, score in zip(id_pairs, labels, scores):
        if label == 'SIMULTANEOUS':
            parents0, scores0 = get_parents(graph, pair[0])
            for index, parent in enumerate(parents0): # copy parents of pair[0] to pair[1]
                if pair[1] not in graph[parent][0]:
                    graph[parent][0].append(pair[1])
                    graph[parent][1].append(score*scores0[index]) # SIMULTANEOUS score * inherited score

            parents1, scores1 = get_parents(graph, pair[1])
            for index, parent in enumerate(parents1):
                if pair[0] not in graph[parent][0]:
                    graph[parent][0].append(pair[0])
                    graph[parent][1].append(score * scores1[index])

            if pair[0] in graph:
                children0, scores0 = graph[pair[0]]
                if pair[1] not in graph:
                    graph[pair[1]] = copy.deepcopy(graph[pair[0]])
                    graph[pair[1]][1] = [score*x for x in graph[pair[1]][1]] # modify scores
                    continue
                for index, child in enumerate(children0):
                    if child not in graph[pair[1]][0]:
                        graph[pair[1]][0].append(child)
                        graph[pair[1]][1].append(score*scores0[index])

            if pair[1] in graph:
                children1, scores1 = graph[pair[1]]
                for index, child in enumerate(children1):
                    if pair[0] not in graph:
                        graph[pair[0]] = copy.deepcopy(graph[pair[1]])
                        graph[pair[0]][1] = [score*x for x in graph[pair[0]][1]] # modify scores
                        continue
                    if child not in graph[pair[0]][0]:
                        graph[pair[0]][0].append(child)
                        graph[pair[0]][1].append(score*scores1[index])

    # remove self-pointing nodes (cycle of one node)
    for u in graph:
        for index, v in enumerate(graph[u][0]):
            if u == v:
                del graph[u][0][index]
                del graph[u][1][index]

    return graph


def modify_tlinks(id_pairs, labels, scores):

    pruned_pairs = []
    before_graph, included_graph, begins_graph, ends_graph= build_graph(id_pairs, labels, scores)

    graph, pairs = prune_tlinks(before_graph)
    pruned_pairs += pairs

    graph, pairs = prune_tlinks(included_graph)
    pruned_pairs += pairs

    graph, pairs = prune_tlinks(begins_graph)
    pruned_pairs += pairs

    graph, pairs = prune_tlinks(ends_graph)
    pruned_pairs += pairs

    # valid_pairs = []
    # for source in before_graph:
    #     for target, score in zip(before_graph[source][0], before_graph[source][1]):
    #         valid_pairs.append((source, target))
    # for source in included_graph:
    #     for target, score in zip(included_graph[source][0], included_graph[source][1]):
    #         valid_pairs.append((source, target))

    count = 0
    for i, label in enumerate(labels):
        if label in DIRECTED and (id_pairs[i] in pruned_pairs):
            labels[i] = 'None'
            count += 1
            print("tlink to modify:", id_pairs[i], label)
    #print "# labels modified:", count

    return labels

def prune_tlinks(G):
    """graph is in format {eid: [[eid_1, eid_2, ...],[score_1, score_2,...]}"""
    graph = copy.deepcopy(G)

    event_nodes = []
    timex_nodes = []
    for k in graph.keys():
        if k[0] == 'e':
            event_nodes.append(k)
        else:
            timex_nodes.append(k)

    # pruned_graph = dict((tid, graph[tid]) for tid in graph.keys() if tid[0] == 't') # start with timex graph
    # try:
    #     assert cycle_exists(simplify_graph(pruned_graph)) == False
    # except AssertionError:
    #     print pruned_graph
    #     print simplify_graph(pruned_graph)
    #     sys.exit("AssertionError")

    pruned_graph = {}
    pruned_pairs = []
    entities = timex_nodes + event_nodes
    for eid in entities:
        candidate_edges = []
        pruned_graph[eid] = graph[eid]

        for node in pruned_graph: # go through each node in pruned graph
            # print "pruned graph:", pruned_graph
            # print "node in pruned_graph:", node
            for target, score in zip(pruned_graph[node][0], pruned_graph[node][1]): # go through the targets of each node
                if eid == target: # a source is found
                    candidate_edges.append(((node, eid), score)) # edge, score of the edge
            for target, score in zip(graph[eid][0], graph[eid][1]): # go through the targets of eid
                if node == target: # a target is found
                    candidate_edges.append(((eid, node), score))  # edge, score of the edge

        #candidate_edges.sort(key=lambda x: x[1])
        permutations = [] # list of list of edeges. every item is a permutations of the same list of edeges
        if candidate_edges:
            candidate_edges.sort(key=lambda x: x[1])
            permutations.append(candidate_edges)  # greedy case
            N = len(candidate_edges)
            limit = min(max(0, N*(N-1)), 10)
            #print "limit for permutations:", limit
            for permu in range(limit):
                temp = copy.copy(candidate_edges)
                np.random.shuffle(temp)
                if temp not in permutations:
                    permutations.append(temp)
            #print "permutations", permutations

            temp_graphs = {}
            sum_scores = {}
            removed_edges = {}
            for permu, candidate_edges in enumerate(permutations):
                temp_graphs[permu] = copy.deepcopy(pruned_graph)
                sum_scores[permu] = 0
                removed_edges[permu] = []
                original_candiates = copy.copy(candidate_edges)
                while cycle_exists(simplify_graph(temp_graphs[permu])) and candidate_edges:
                    candidate = candidate_edges.pop(0)
                    sum_scores[permu] += candidate[1]
                    source = candidate[0][0]
                    target = candidate[0][1]

                    ## There seems to be a weird bug here.
                    ## sometimes an edge is found in the pruned graph, but it does not show up in temp_graps[permu]
                    try:
                        index = temp_graphs[permu][source][0].index(target)
                    except ValueError:
                        print(temp_graphs[permu][source])
                        print(source, target)
                        print("candidate edges:", original_candiates)
                        print("removed edges:", removed_edges[permu])
                        sys.exit("Value Error")

                    temp_graphs[permu][source][0].pop(index) # remove from id list
                    temp_graphs[permu][source][1].pop(index) # remove from score list
                    removed_edges[permu].append((source, target))
                    removed_edges[permu].append((target, source))
                    # if temp_graphs[permu].get('t3', '') != pruned_graph.get('t3', ''):
                    #     print "t3 changed!!!"
                    #     print permu
                    #     print candidate
                    #     print temp_graphs[permu].get('t3', '')
                    #     print pruned_graph.get('t3', '')

            # print "\nsum_scores", sum_scores
            min_permu = min(sum_scores, key=sum_scores.get)
            pruned_graph = temp_graphs[min_permu]
            pruned_pairs += removed_edges[min_permu]

    # print "pruned pairs", pruned_pairs

    return pruned_graph, pruned_pairs








