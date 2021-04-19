import os
import subprocess as proc


def solvePBQP(pbqp_path, nodes, edges, peo):
    write_lines("temp.txt", export_graph(nodes, edges, peo))
    proc.run(f"{pbqp_path} ./temp.txt ./temp_out.txt")

    solution = read_solution("temp_out.txt")
    return get_cost(nodes, edges, solution), solution
    

def get_cost(nodes, edges, solution):
    cost = 0
    
    for i, node in zip(solution, nodes):
        cost += node[i]
        
    for edge in edges:
        s, t, costs = edge
        s_size = len(nodes[s])
        
        s = solution[s]
        t = solution[t]
        
        cost += costs[s * s_size + t]
        
    return cost
    

def export_graph(nodes, edges, peo):
    data = [f"{len(nodes)} {len(edges)}"]
    
    for node in nodes:
        size = len(node)
        node = ' '.join(list(map(str, node)))

        data.append(f"{size} {node}")
                    
    for edge in edges:
        source, target, edge = edge
        size = len(edge)
        edge = ' '.join(list(map(str, edge)))

        data.append(f"{source} {target} {size} {edge}")
                    
    peo = list(map(str, peo))
    data.append(' '.join(peo))
                      
    return data
       

def write_lines(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")
        

def read_solution(path):
    with open(path, "r") as f:
        solution = f.readlines()[0].split(" ")[:-1]
        solution = list(map(int, solution))
                    
    return solution


def build_straight_NN_graph(cost_model, layer_info):
    nodes = []
    edges = []
    
    node_names = list(cost_model[0].keys())
    trans_stoi = dict(map(lambda x: (x[1], x[0]), enumerate(node_names[-9:])))
    
    for layer in cost_model:
        costs = np.array(list(layer.values())[:-9])
        costs[costs == 0] = -1
        costs = list(map(int, list(costs)))
        
        nodes.append(costs)
        
    for layer in cost_model:
        costs = np.array(list(layer.values())[-9:])
        edge = []
        
        for source in range(len(nodes[0])):
            for target in range(len(nodes[1])):
                source_shape = formats[formats["prim"] == node_names[source]]["out"][0]
                target_shape = formats[formats["prim"] == node_names[target]]["inn"][0]
        
                edge.append(costs[trans_stoi[source_shape + "-to-" + target_shape]])

    return nodes, edges, list(range(len(nodes)))
