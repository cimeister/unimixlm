import educative.course1.stacks_queues.stack as s
import educative.course1.graphs.graph as g

input_num_vertices = 5
input_edges = {0: [1, 2], 1: [3, 4]}
expected_output = "02143 or 02134 or 01432 or 01342"


# this code implements Depth First Traversal in a graph. Each element in the graph's adjacency list
# represents a list of nodes directly connected to the current element. Each of these lists represent
# levels in the graph. We traverse all the nodes from root node to last node first, then we back-track
# to the last visited common node and then we take second branch from there, until we back-track to root node
# and no branches are left unvisited.
# --------------
# For traversal:
# --------------
# 1. We first push the root node in a stack and mark it visited.
# 2. Now we get the list of adjacent nodes from the root node and push all adjacent nodes in the list
#    to the stack.
# 3. As we are pushing the adjacent nodes to the stack, we mark them as visited.
# 4. Simultaneously, we pop the nodes from the stack and append the node's value to the result
# --------------
def bfs_traversal(graph, root):
    result = ""
    visited = [False] * graph.num_vertices

    stack = s.Stack(graph.num_vertices, True) # suppress_printing = True
    stack.push(root)

    while not stack.is_empty():
        g_node = stack.pop()
        result += str(g_node)

        dll = graph.adjacency_list[g_node]
        if dll is not None:
            current = dll.head
            while current:
                if not visited[current.value]:
                    stack.push(current.value)
                    visited[current.value] = True
                current = current.next

    return result


def main():
    # create a graph from input data
    graph = g.Graph(input_num_vertices, True) # suppress_printing = True
    for x in input_edges.keys():
        for y in input_edges[x]:
            graph.add_edge(x, y)

    # set a starting point for traversal
    root = 0

    print("Input:")
    print("Graph in Adjacency List Representation:")
    print(str(graph.prettify()))

    print("Expected: " + str(expected_output))
    print("Output: " + str(bfs_traversal(graph, root)))


if __name__ == '__main__':
    main()