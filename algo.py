import numpy as np
import scipy.io
import networkx as nx
import matplotlib.pyplot as plt

def preprocess_and_read_graph(file_path):
    """
    Reads and preprocesses a graph from a Matrix Market (.mtx) file.
    If the matrix is not square, it converts it into a square adjacency matrix.
    """
    matrix = scipy.io.mmread(file_path)

    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        print("Matrix is not square. Converting to square adjacency matrix...")
        # Create a square matrix by adding zero rows/columns
        size = max(matrix.shape)
        square_matrix = np.zeros((size, size))
        square_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
        matrix = square_matrix

    # Convert to a NetworkX graph
    graph = nx.from_numpy_array(matrix)
    return graph

def fiduccia_mattheyses(graph):
    """
    Implements the Fiduccia–Mattheyses (FM) algorithm for the minimum bisection problem.
    Ensures that the partitions are as equal as possible.
    """
    # Initial partition: split nodes into two equal sets
    nodes = list(graph.nodes)
    half_size = len(nodes) // 2
    partition_a = set(nodes[:half_size])
    partition_b = set(nodes[half_size:])

    # Log initial partition sizes
    print(f"Initial partition sizes: A={len(partition_a)}, B={len(partition_b)}")

    # Ensure partitions are balanced
    while len(partition_a) > len(partition_b):
        node_to_move = partition_a.pop()
        partition_b.add(node_to_move)

    # Log balanced partition sizes
    print(f"Balanced partition sizes: A={len(partition_a)}, B={len(partition_b)}")

    # Calculate initial cut size
    cut_size = calculate_cut_size(graph, partition_a, partition_b)
    print(f"Initial cut size: {cut_size}")

    # FM algorithm main loop
    for _ in range(len(nodes)):
        # Calculate gain for each node
        gains = {}
        for node in nodes:
            if node in partition_a:
                gains[node] = calculate_gain(graph, node, partition_a, partition_b)
            else:
                gains[node] = calculate_gain(graph, node, partition_b, partition_a)

        # Find the node with the maximum gain
        max_gain_node = max(gains, key=gains.get)
        if max_gain_node in partition_a and len(partition_a) > len(partition_b):
            partition_a.remove(max_gain_node)
            partition_b.add(max_gain_node)
        elif max_gain_node in partition_b and len(partition_b) > len(partition_a):
            partition_b.remove(max_gain_node)
            partition_a.add(max_gain_node)

        # Recalculate cut size
        cut_size = calculate_cut_size(graph, partition_a, partition_b)
        print(f"Updated cut size: {cut_size}")

        # Log partition sizes after each iteration
        print(f"Partition sizes after iteration: A={len(partition_a)}, B={len(partition_b)}")

    return partition_a, partition_b

def calculate_cut_size(graph, partition_a, partition_b):
    """
    Calculates the cut size between two partitions.
    """
    cut_edges = 0
    for edge in graph.edges:
        if (edge[0] in partition_a and edge[1] in partition_b) or (edge[0] in partition_b and edge[1] in partition_a):
            cut_edges += 1
    return cut_edges

def calculate_gain(graph, node, source_partition, target_partition):
    """
    Calculates the gain of moving a node from the source partition to the target partition.
    """
    external_cost = sum(1 for neighbor in graph.neighbors(node) if neighbor in target_partition)
    internal_cost = sum(1 for neighbor in graph.neighbors(node) if neighbor in source_partition)
    return external_cost - internal_cost

if __name__ == "__main__":
    # Path to the add20_b.mtx file
    file_path = "/Users/venoralph/GSU_PhD/Advance_Algorithms/Project/add20/add20_b.mtx"

    # Preprocess and read the graph
    graph = preprocess_and_read_graph(file_path)

    # Run the FM algorithm
    partition_a, partition_b = fiduccia_mattheyses(graph)

    # Output the results
    print("Partition A:", partition_a)
    print("Partition B:", partition_b)

    # Count the nodes in each partition
    count_a = len(partition_a)
    count_b = len(partition_b)
    print(f"Number of nodes in Partition A: {count_a}")
    print(f"Number of nodes in Partition B: {count_b}")

    # Visualize the partition sizes
    plt.bar(['Partition A', 'Partition B'], [count_a, count_b], color=['blue', 'orange'])
    plt.title('Partition Sizes')
    plt.ylabel('Number of Nodes')
    plt.show()