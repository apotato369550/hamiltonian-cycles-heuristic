import random

# function creates complete graph given:
# edges - number of edges as an int
# weight_range - the range of weights as a tuple
# replace - whether to sample the weights with or without replacement
def generate_complete_graph(vertices, weight_range=(1, 10), replace=False):
    # create n * n graph to be returned
    graph = [[0] * vertices for _ in range(vertices)]

    # if weights are to be assigned with replacement
    if replace:
        for i in range(vertices):
            for j in range(i + 1, vertices):
                # asterisk unpacks the tuple
                weight = random.randint(*weight_range)
                graph[i][j] = graph[j][i] = weight
    else:
        # create list of possible weights
        possible_weights = list(range(weight_range[0], weight_range[1] + 1))
        total_edges = vertices * (vertices - 1) // 2
        if len(possible_weights) < total_edges:
            raise ValueError(f"Not enough unique weights for the number of edges. Must need at least {total_edges} total edges for graph with {vertices} vertices")
    return graph