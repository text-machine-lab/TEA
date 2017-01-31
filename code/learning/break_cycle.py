import copy

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

    for pair, label, score in zip(id_pairs, labels, scores):
        if label in ('BEFORE', 'IBEFORE'):
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

    return before_graph, included_graph


def modify_tlinks(id_pairs, labels, scores):

    before_graph, included_graph = build_graph(id_pairs, labels, scores)
    before_graph, pruned_b_pairs = prune_tlinks(before_graph)
    included_graph, pruned_i_pairs = prune_tlinks(included_graph)

    pruned_pairs = pruned_b_pairs + pruned_i_pairs
    print pruned_pairs

    # valid_pairs = []
    # for source in before_graph:
    #     for target, score in zip(before_graph[source][0], before_graph[source][1]):
    #         valid_pairs.append((source, target))
    # for source in included_graph:
    #     for target, score in zip(included_graph[source][0], included_graph[source][1]):
    #         valid_pairs.append((source, target))

    count = 0
    for i, label in enumerate(labels):
        if label in ('BEFORE', 'IBEFORE', 'AFTER', 'IAFTER', 'IS_INCLUDED', 'INCLUDES') and (id_pairs[i] in pruned_pairs):
            labels[i] = 'None'
            count += 1
            print "tlink to modify:", id_pairs[i], label
    print "# labels modified:", count

    return labels

def prune_tlinks(G):
    """graph is in format {eid: [[eid_1, eid_2, ...],[score_1, score_2,...]}"""
    graph = copy.deepcopy(G)

    event_nodes = sorted([x for x in graph.keys() if x[0] == 'e'], key=lambda i: i[1:])

    pruned_graph = dict((tid, graph[tid]) for tid in graph.keys() if tid[0] == 't') # start with timex graph
    assert cycle_exists(simplify_graph(pruned_graph)) == False

    pruned_pairs = []
    for eid in event_nodes:
        candidate_edges = []

        pruned_graph[eid] = graph[eid]
        # print pruned_graph, graph
        # print '__________'
        # if cycle_exists(simplify_graph(pruned_graph)):
            # find all nodes pointing to the source node of the new edge

        for node in pruned_graph: # go through each node in pruned graph
            # print "pruned graph:", pruned_graph
            # print "node in pruned_graph:", node
            for target, score in zip(pruned_graph[node][0], pruned_graph[node][1]): # go through the targets of each node
                # print "node, target, score", node, target, score
                if eid == target: # a source is found
                    candidate_edges.append(((node, eid), score)) # edge, score of the edge
            for target, score in zip(graph[eid][0], graph[eid][1]): # go through the targets of eid
                # print "eid, target, score", eid, target, score
                if node == target: # a target is found
                    candidate_edges.append(((eid, node), score))  # edge, score of the edge

        candidate_edges.sort(key=lambda x: x[1])
        while cycle_exists(simplify_graph(pruned_graph)) and candidate_edges:
            candidate = candidate_edges.pop(0)
            source = candidate[0][0]
            target = candidate[0][1]
            index = pruned_graph[source][0].index(target)
            pruned_graph[source][0].pop(index) # remove from id list
            pruned_graph[source][1].pop(index) # remove from score list
            pruned_pairs.append((source, target))
            pruned_pairs.append((target, source)) # both pairs should be removed

    return pruned_graph, pruned_pairs








