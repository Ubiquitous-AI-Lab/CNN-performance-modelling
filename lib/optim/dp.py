import numpy as np
import pandas as pd

import torch
import random


def trace(path, u, v, res=[]):
    if u == v:
        return [v]
    else:
        return trace(path, u, path[(u, v)]) + [v]


def sp(graph, u, v, path, results={}):    
    if u == v:
        return 0
    
    if (u, v) not in results:
        if v in graph:
            nodes = graph[v]
            costs = [sp(graph, u, s[0], path, results) + s[1] for s in nodes]
            s = np.argmin(costs)

            results[(u, v)] = costs[s]
            path[(u, v)] = nodes[s][0]
        else:
            results[(u, v)] = np.inf
            path[(u, v)] = None
            
    return results[(u, v)]


def shortest_path(graph, u, v):
    path = {}
    score = sp(graph, u, v, path, {})
    return score, trace(path, u, v)


def build_test_graph(depth, breadth, weight_min, weight_max):
    total = depth * breadth
    graph = {i:[] for i in range(-1, total+1)}

    for node in list(range(breadth)):
        graph[node].append((-1, 0))
    
    for node in list(range(total)):
        if node < total - breadth:
            start = node//breadth + 1
            for other in list(range(start * breadth, (start + 1) * breadth)):
                graph[other].append((node, random.randint(weight_min, weight_max)))
        else:
            graph[total].append((node, 0))
                
    return graph


def build_conv_graph(layer_data, cost_model):
    depth = len(cost_model)
    
    formats = ["chw", "hcw", "hwc"]
    
    graph = {"end": []}
    key_store = [[]]

    for node in range(len(layer_data)):
        layer = layer_data.iloc[node]
        cost = cost_model[0][layer.prim] if layer.prim in cost_model[0] else 0
        
        if cost > 0:
            key = "0::" + layer.prim + "::none::" + layer.out
            graph[key] = [("start", cost)]
            key_store[0].append(key)
        
        for layout in formats:
            graph["end"].append((str(depth-1) + "::" + layer.prim + "::" + layout + "::" + layer.out, 0))
        
    for i in range(1, depth):
        key_store.append([])
        
        for node in range(len(layer_data)):
            layer = layer_data.iloc[node]
            cost = cost_model[i][layer.prim] if layer.prim in cost_model[i] else 0

            if cost > 0:
                for layout in formats:
                    edges = []
                    
                    for key in key_store[i-1]:
                        if key.split("::")[-1] == layout:
                            in_layout = layer.inn
                            cost_transform = cost_model[i][f"{layout}-to-{in_layout}"]
                            edges.append((key, cost_transform + cost))
                            
                    if len(edges) > 0:
                        key = f"{i}::{layer.prim}::{layout}::{layer.out}"
                        graph[key] = edges
                        key_store[i].append(key)
                            
    return graph