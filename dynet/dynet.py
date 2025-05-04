#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:32:28 2023

This module is used for network identification and identifiability analysis

@author: edumapu
"""
import numpy as np
import sympy as sym
from scipy.linalg import null_space
from collections import defaultdict
import itertools
from math import comb
from random import choice

__all__ = ['digraph', 'net_create_sym_variables']

# Create a digraph class
class digraph():
    def __init__(self, nodes, edges, adj = None):
        self.n = len(nodes)
        # if self.n > 0:
        self.nodes = nodes
        self.nodes_enum = dict(zip(nodes, range(self.n)))
        self.set_nodes = set(nodes)
        if adj == None:
            self.edges = edges
            # Get the Adjacency matrix from edges
            self.get_adjacency()
        else:
            self.A = np.asarray(edges)
            # Get edges from the Adjacency matrix
            self.get_edges()

        # Find the other types of nodes
        self.find_dsources_dsinks()
        # Get list of out-neighbors
        self.find_outneighbors_net()
        # Set list of in-neighbors
        self.find_inneighbors_net()
    
    def __add__(self, other):
        """ Define an addition of dipghaphs"""
        nodes = list(set(self.nodes).union(set(other.nodes)))
        edges = list(set(self.edges).union(set(other.edges)))
        return digraph(nodes, edges)
    
    def __str__(self):
        return f'Digraph model with {self.nodes} nodes and edges\n {self.edges}'
    
    def __repr__(self):
        return f'digraph({self.nodes}, {self.edges})'
    
    def add_edge(self, edge):
        """ Add an edge to the graph """
        if not edge[0] in self.set_nodes:
            # Add vertex
            self.nodes.append(edge[0])
            # Update 
            self.set_nodes = set(self.nodes)
        if not edge[1] in self.set_nodes:
            # Add vertex
            self.nodes.append(edge[1])
            # Update 
            self.set_nodes = set(self.nodes)
        # Add edge to the set of edges
        # Check if this edge is not in the edges
        if edge not in self.edges:
            self.edges.append(edge)
            # Update the number of nodes
            self.n = len(self.nodes)
            # Update the Adjacency matrix
            self.get_adjacency()

    def find_dsources_dsinks(self):
        """ Find sources, sinks, dources, and dinks """
        self.sources = []
        self.sinks = []
        self.dources = []
        self.dinks = []
        for j in range(self.n):
            # Check column
            if (sum(self.A[:, j]) == 0):
                self.sinks.append(self.nodes[j])
                continue
            if (sum(self.A[j, :]) == 0):
                self.sources.append(self.nodes[j])
                continue
            # Check dource
            Npj = np.where(self.A[:, j] == 1)[0]
            Nmj = np.where(self.A[j, :] == 1)[0]
            ANpjNmj = self.A[np.ix_(Npj, Nmj)]

            for k in range(len(Nmj)):
                if all(ANpjNmj[:, k]):
                    self.dinks.append(self.nodes[j])
                    break
            for k in range(len(Npj)):
                if all(ANpjNmj[k, :]):
                    self.dources.append(self.nodes[j])
                    break
    
    def check_node(self, node):
        """ Check if the node is present """
        if node in self.nodes:
            return True
        else:
            return False
    
    def check_edge(self, edge):
        """ Check if the edge is present """
        if edge in self.edges:
            return True
        else:
            return False
                
    # TODO this is not an efficiet way to treat this problem: check another way out
    def find_induced_tree(self, node):
        """ Find the induce tree from node """
        
        # Set of nodes
        V = []
        # Set of edges
        E = []
        # Append the first node
        V.append(node)
        
        
        for i in self.nodes:
            if i != node:
                # Mark all vertices as not visited
                visited = [False]*self.n
                # Initialize paths
                paths = []
                path = []
                # Call recursive routine
                self.find_paths_from(node, i, visited, path, paths)
                # Check if there is more than a path
                if len(paths) == 1:
                    # Add this edge to the network
                    path = paths[0]
                    for k in range(len(path)-1):
                        edge = (path[k], path[k+1])
                        if edge not in E:
                            E.append((path[k], path[k+1]))
                        # Check if they are part of the verices
                        if path[k] not in V:
                            # Add it to V
                            V.append(path[k])
                        if path[k+1] not in V:
                            V.append(path[k+1])
                # Get all paths from node to i
        
        return digraph(V, E) 
    
    def find_remaining_edges_PPN(self, T):
        """ Find all PPNs from the difference of nodes"""
        # Get the edges difference
        Eo = set(self.edges)
        Ec = set(T.edges)
        
        # Check which nodes are in Eo that are not in Ec
        Ebar = list(Eo.difference(Ec))
        # List of PPNs paths
        PPNs = []
        # Position of the next edge to be tested
        pos_list = [0]
        # While all nodes are not identifiable
        while len(Ebar) != 0:
            self.find_remaining_edges_PPN_rec(pos_list, Ebar, PPNs)
        
        return PPNs
        
    
    def find_remaining_edges_PPN_rec(self, pos_list, Ebar, PPNs):
        """ Find the remaining edges using parallel paths """
        
        # Get the first edge
        e = Ebar[pos_list[0]]
        # Get vertices of edge e
        i, j = e
        # Check how many paths there are
        paths = self.find_all_paths_i_to_j(j, i)
        # Verify how many unknown edges are in the paths
        Pe = set(paths.edges)
        Ed = list(Pe.intersection(Ebar))
        if len(Ed) < 2:
            # Find a PPN for it
            paths_list = self.find_all_paths_i_to_j_(j, i)
            # check the longest path
            maxpath = paths_list[np.argmax([len(path) for path in paths_list])]
            # Propose a PPN
            ppnpaths = [maxpath, Ed[0]]
            # Convert to 
            PPNs.append(self.paths_to_graph(ppnpaths))
            # Remove edge from Ebar
            Ebar.remove(e)
            pos_list[0] = 0
        else:
            # Try another node in Ebar
            # Roll
            pos_list[0] += 1
            if pos_list[0] > len(Ebar):
                pos_list[0] = 0
            # Ebar = shift_list(Ebar)
            # self.find_remaining_edges_PPN_rec(Ebar, PPNs)
        
        return 1
    
    def list_all_induced_trees(self):
        """ List all the induced trees """
        L = []
        
        for node in self.nodes:
            L.append(self.find_induced_tree(node))
        
        return L
    
    def find_induced_tree_cover(self, randomized=False):
        """ Find an induced tree cover """
        # Get the list of induced trees
        iTs = self.list_all_induced_trees()
        # Nodes selected by the induced trees
        iTsNodes = []
        # Get a copy 
        # Initialize an empty digraph
        Go = digraph([], [])
        # Initialize the 
        # Start to m
        # Pre select the induced trees
        Remove = []
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    # Check if i is contained in j
                    iti = iTs[i]
                    itj = iTs[j]
                    # Check if itj is contained in iti
                    if iti.check_contained(itj):
                        # If is contained, then removed itj
                        Remove.append(j)
                    if itj.check_contained(iti):
                        # If is contained, them remove iti
                        Remove.append(i)
        
        # Get only the selected nodes
        Selection = [i for i in list(range(self.n)) if i not in Remove]
        
        #
        ITsedges = defaultdict(list)
        Sel = []
        addedges = []
        
        # Make the GT
        for edge in self.edges:
            # How many ITs have this edge?
            for i in Selection:
                t = iTs[i]
                if t.check_edge(edge):
                    ITsedges[edge].append(i)
            # If only one IT has
            hmn = len(ITsedges[edge])
            if hmn == 1:
                # ind = self.nodes[ITsedges[edge][0]]
                ind = ITsedges[edge][0]
                if ind not in Sel: 
                    Sel.append(ind)
            if hmn > 1:
                addedges.append(edge)
        
        # Order It
        Sel.sort()
        # Check if there are iTs covering the nodes
        for edge in addedges:
            # check edge
            if all([i not in Sel for i in ITsedges[edge]]):
                # Check the biggest
                sel = [len(iTs[i].edges) for i in ITsedges[edge]]
                # Get the maximum index
                Sel.append(sel.index(max(sel)))
        
        # Now update IT nodes
        iTsNodes = []
        for i in Sel:
            iTsNodes.append(self.nodes[i])
            t = iTs[i]
            for edge in t.edges:
                Go.add_edge(edge)
        
        return Go, iTsNodes
    
    def Michel_algorithm(self, randomized=False):
        """ Make an EMP following Michel's algorithm """
        # 
        B = set()
        C = set()
        # Get the list of induced trees
        iTs = self.list_all_induced_trees()
        # Nodes selected by the induced trees
        iTsNodes = []
        # Get a copy 
        # Initialize an empty digraph
        Go = digraph([], [])
        # Initialize the 
        # Start to m
        # Pre select the induced trees
        Remove = []
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    # Check if i is contained in j
                    iti = iTs[i]
                    itj = iTs[j]
                    # Check if itj is contained in iti
                    if iti.check_contained(itj):
                        # If is contained, then removed itj
                        Remove.append(j)
                    if itj.check_contained(iti):
                        # If is contained, them remove iti
                        Remove.append(i)
        
        # Get only the selected nodes
        Selection = [i for i in list(range(self.n)) if i not in Remove]
        
        #
        ITsedges = defaultdict(list)
        Sel = []
        addedges = []
        
        # Make the GT
        for edge in self.edges:
            # How many ITs have this edge?
            for i in Selection:
                t = iTs[i]
                if t.check_edge(edge):
                    ITsedges[edge].append(i)
            # If only one IT has
            hmn = len(ITsedges[edge])
            if hmn == 1:
                # ind = self.nodes[ITsedges[edge][0]]
                ind = ITsedges[edge][0]
                if ind not in Sel: 
                    Sel.append(ind)
            if hmn > 1:
                addedges.append(edge)
        
        # Order It
        Sel.sort()
        # Check if there are iTs covering the nodes
        for edge in addedges:
            # check edge
            if all([i not in Sel for i in ITsedges[edge]]):
                # Check the biggest
                sel = [len(iTs[i].edges) for i in ITsedges[edge]]
                # Get the maximum index
                Sel.append(sel.index(max(sel)))
        
        # Now update IT nodes
        iTsNodes = []
        for i in Sel:
            iTsNodes.append(self.nodes[i])
            t = iTs[i]
            # Add the sources
            B.update(t.sources)
            # Add the sinks
            C.update(t.sinks)
            # For every node that is neither a source nor a sink
            for node in t.nodes:
                # Check the out-neighbors of a node
                if node not in t.sources and node not in t.sinks:
                    # Check if it there are outneighbors
                    if len(set(self.outneighbors[node])-\
                           set(t.nodes)) > 0:
                        # Excite this node
                        B.update([node])
                    # Check if there are inneighbors 
                    if len(set(self.inneighbors[node])-\
                           set(t.nodes)) > 0:
                        # Measure the node
                        C.update([node])
                
        return B, C, iTsNodes

    
    def check_contained(self, other):
        """ Check if the other graph is contained in self """
        # Convert to sets
        V1 = set(self.nodes)
        E1 = set(self.edges)
        V2 = set(other.nodes)
        E2 = set(other.edges)
        # Check if V2 is contained in V1 and if E2 is contained in E1
        if E2.issubset(E1) and V2.issubset(V1):
            return True
        else:
            return False
                
    
    def find_tree_cover(self):
        """Find the tree cover of the digraph"""
        
        
        return NotImplemented

    def get_adjacency(self):
        """ Get the adjacency matrix from the list of edges """
        A = np.zeros((self.n, self.n))
        for edge in self.edges:
            # Get index of vertices
            if (edge[0] not in self.nodes) or (edge[1] not in self.nodes):
                raise ValueError('Element in edges not in vertex set')
            A[self.nodes.index(edge[1])][self.nodes.index(edge[0])] = 1
        self.A = A

    def get_edges(self):
        """ Get the edges from adjacency matrix """
        edges = []
        for j in range(self.n):
            for i in range(self.n):
                if (self.A[j, i] == 1):
                    edges.append((self.nodes[i], self.nodes[j]))

        self.edges = edges
    
    def find_inneighbors_net(self):
        """ Function to find all in-neighbors of all nodes"""
        self.inneighbors = defaultdict(list)
        
        for j in range(self.n):
            # Get in-neighbors
            N = np.nonzero(self.A[j, :])[0].tolist()
            for n in N:
                self.inneighbors[self.nodes[j]].append(self.nodes[n])

    
    def find_outneighbors_net(self):
        """Function to find all out-neighbors of all nodes"""
        self.outneighbors = defaultdict(list)

        for j in range(self.n):
            # Get out-neighbors
            N = np.nonzero(self.A[:, j])[0].tolist()
            for n in N:
                self.outneighbors[self.nodes[j]].append(self.nodes[n])
    
    def paths_to_graph(self, paths):
        """Transform a given path to a graph """
        vertex_union = set()
        edges_union = set()
        for path in paths:
            vertex_union = vertex_union.union(set(path))
            edges_union = edges_union.union(set(zip(path[0:-1], path[1:])))
        
        return digraph(list(vertex_union), list(edges_union))
        
        
    def find_paths_from(self, u, d, visited, path, paths):
        # Mark the current node as visited
        visited[self.nodes_enum[u]] = True
        path.append(u)
        
        if u == d:
            paths.append(path.copy())
        else:
            for i in self.outneighbors[u]:
                if visited[self.nodes_enum[i]] == False:
                    self.find_paths_from(i, d, visited, path, paths)
        
        # Remove current vertex from path and mark it as unvisited
        path.pop()
        visited[self.nodes_enum[u]] = False
                    
    def find_flow(self, j, i):
        """ Returns all paths that go from node i to node j """
        # Mark all vertices as not visited
        visited = [False]*self.n
        # Initialize paths
        paths = []
        path = []
        # Call recursive routine
        self.find_paths_from(i, j, visited, path, paths)
        # Return a graph from it
        return self.paths_to_graph(paths)

    def find_all_paths_i_to_j(self, j, i):
        """ Returns all paths that go from node i to node j """
        # Mark all vertices as not visited
        visited = [False]*self.n
        # Initialize paths
        paths = []
        path = []
        # Call recursive routine
        self.find_paths_from(i, j, visited, path, paths)
        # Return a graph from it
        return self.paths_to_graph(paths)
    
    def find_all_paths_i_to_j_(self, j, i):
        """ Returns all paths that go from node i to node j """
        # Mark all vertices as not visited
        visited = [False]*self.n
        # Initialize paths
        paths = []
        path = []
        # Call recursive routine
        self.find_paths_from(i, j, visited, path, paths)
        # Return a graph from it
        return paths

    
    def find_ppn(self, edge):
        """ Find the PPNs from node i to j"""
        # Find all paths from i to j
        # paths = self.find_all_paths_i_to_j(edge[1], edge[0])
        paths = self.find_all_paths_i_to_j_(edge[1], edge[0])
        # Sort paths by lenght
        paths.sort(key=len)
        simple = True
        # Nodes involved in the paths
        V = set()
        # Iterate over the paths
        for path in paths:
            # Iterate over the nodes
            for node in path:
                if node not in V:
                    # Add to the set of nodes
                    V.add(node)
        # Check if the paths are simple
        for path in paths:
            # Check the lenght of the path
            # TODO: The PPN should be previously defined such that we only
            # analyze the right nodes not pertaining to the PPN
            if len(path) > 2:
                for i in path[1:-1]:
                    # For every node in path check if there is multiple paths
                    apaths = self.find_all_paths_i_to_j_(edge[1], i)
                    # Sort paths by lenght
                    apaths.sort(key=len)
                    # If there is multiple paths, then it is not simple
                    if len(apaths) > 1:
                        simple = False
        # Check if exists a set of paths forming a PPN
        # Take the shortest paths
        ppn_paths = paths[:2]
        # Transform into grahs
        ppn = self.paths_to_graph(ppn_paths)
        
        return ppn, simple

    
    def DFS(self, v, discovered, departure, time):
        """Perform depth first search on the graph and set departure time of all vertices"""
        # Mark the current node as discovered
        discovered[self.nodes_enum[v]] = True
        # For every (v, u)
        for u in self.outneighbors[v]:
            if not discovered[self.nodes_enum[u]]:
                time = self.DFS(u, discovered, departure, time)
        # Ready to backtrack
        departure[self.nodes_enum[v]] = time
        time += 1
        
        return time
    
    def BFS(self, s, t, parent):
        """ Perform breath first search for capacity network """
        # Mark all the vertices as not visited
        visited = [False]*(self.n)
        # Create a queue for BFS
        queue = []
        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[self.nodes.index(s)] = True
        # Standard BFS Loop
        while queue:
            # Dequeue a vertex from queue and print it
            u = queue.pop(0)
            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.C[self.nodes.index(u)]):
                if visited[ind] == False and val > 0 :
                    queue.append(self.nodes[ind])
                    visited[ind] = True
                    parent[ind] = u
        # If we reached sink in BFS starting from source, then return
        # true, else false
        return True if visited[self.nodes.index(t)] else False

    
    # Ford Fulkeson algorithm
    def FordFulkerson(self, source, sink):
        """ Ford Fulkesson algorithm to find maximum flow """
		# This array is d by BFS and to store path
        parent = [-1]*(self.n)
        max_flow = 0 # There is no flow initially
		# Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent):
            # Find minimum residual capacity of the edges along the
			# path filled by BFS. Or we can say find the maximum flow
			# through the path found.
            path_flow = float("Inf")
            s = sink
            while(s != source):
                path_flow = min(path_flow, self.C[self.nodes.index(parent[self.nodes.index(s)])]
                                                 [self.nodes.index(s)])
                s = parent[self.nodes.index(s)]
            # Add path flow to overall flow
            max_flow += path_flow
            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while(v != source):
                u = parent[self.nodes.index(v)]
                self.C[self.nodes.index(u)][self.nodes.index(v)] -= path_flow
                self.C[self.nodes.index(v)][self.nodes.index(u)] += path_flow
                v = parent[self.nodes.index(v)]

        return max_flow
    
    
    def isDAG(self):
        """ Check whether the graph is a DAG"""
        # Set all nodes as undiscovered
        discovered = [False]*self.n
        # Keep track of departure time
        departure = [None]*self.n
        # Reset time
        time = 0
        # Perform DFS transversal from all undiscovered nodes
        for i in range(self.n):
            if not discovered[i]:
                time = self.DFS(i, discovered, departure, time)
        # Check if the given directed graph is DAG or not
        for u in range(self.n):
            # Check if (u, v) forms a back edge
            for v in self.outneighbors[u]:
                if departure[u] <= departure[v]:
                    return False
        # No back edges
        return True
    
    # Generate all EMPs with cardinality n
    def generate_emps_card_k(self, k):
        """ Generate all EMPs with cardinality k"""
        # cEMPs = list(itertools.combinations(list(range(2*self.n)), k))
        # Make a dictionary for sources and dinks
        sources = self.nodes_enum[self.sources]
        sinks   = self.nodes_enum[self.sinks]
        dources = self.nodes_enum[self.dources]
        dinks   = self.nodes_enum[self.dinks]
        D = {}
        # Sources and Dources must be excited
        for source in sources:
            D.update((source, [1]))
        for dource in dources:
            D.update((dource, [1]))
        # Sinks and dinks must be measured
        for sink in sinks:
            D.update((sink+self.n, [1]))
        for dink in dinks:
            D.update((dink+self.n, [1]))
        
        yield itertools.product(*[D.get(i, [0, 1]) for i in range(2*self.n)])
        
    def find_all_EMPs_Antonie(self, k):
        """ Check all EMPs identifiable by Antoine's algorithm """
        nemps = 0
        cont = 0
        cont_valid = 0
        EMPs = []
        # Find the sources, and dources
        F  = []
        S  = []
        Df = []
        Ds = []
        for source in self.sources:
           F.append(self.nodes_enum[source]) 
        for sink in self.sinks:
            S.append(self.nodes_enum[sink])
        for dource in self.dources:
            Df.append(self.nodes_enum[dource])
        for dink in self.dinks:
            Ds.append(self.nodes_enum[dink])
        # All sources and dources must be excited
        nf = len(F) + len(Df)
        # All sinks and dinks must be measured
        ns = len(S) + len(Ds)
        # Map of the signals
        L = list(range(2*self.n))
        # Remove excitation from sources and dources
        L = [i for i in L if i not in F]
        L = [i for i in L if i not in Df]
        L = [i for i in L if i not in [s+self.n for s in S]]
        L = [i for i in L if i not in [ds+self.n for ds in Ds]]
        # All combinations 
        combs = combinations(L, k-nf-ns)
        # Inverted map
        inv_map = {v: k for k, v in self.nodes_enum.items()}
        # Iterate over the combinations
        for emp in combs:
            cont += 1
            # Add sources and dources to B    
            B = F + Df
            # Add sinks and dinks to C
            C = S + Ds
            # Add another elements
            for x in emp:
                if x >= self.n:# - nf:
                    C.append(x-self.n)
                else:
                    B.append(x)
            # Check if C is empty or B is empty
            if B != [] and C != []:
                # Check if at least all nodes are either excited or measured
                if (set(list(range(self.n)))).issubset(set(B).union(set(C))):
                    # Check Identifiability of B and C
                    out, i_edges, ni_edges = AntoineId(self.A, B, C, nsamples=3)
                    cont += 1
                    if out:
                        # Convert to original nodes
                        Bo = []
                        Co = []
                        for b in B:
                            Bo.append(inv_map[b])
                        for c in C:
                            Co.append(inv_map[c])
                        # Tranform into sets
                        Bo = set(Bo)
                        Co = set(Co)
                        # Print and Store
                        EMPs.append((Bo, Co))
                        cont_valid += 1
                        print(f"Valid EMP {cont_valid} found: {(Bo, Co)}")
                        nemps += 1

        return EMPs
        
    def check_identifiability_antoine(self, B, C):
        """ The sets of identifiability """
        # Convert the sets to proper order
        Bordered = [self.nodes_enum[ex] for ex in B]
        Cordered = [self.nodes_enum[me] for me in C]
        # Check if there is an adjacency matrix
        # Check identifiability
        return AntoineId(self.A, Bordered, Cordered, nsamples=3)
        

class FlowGraph(digraph):
    def __init__(self, nodes, edges, C, adj=None):
        digraph.__init__(self, nodes, edges, adj)
        # Add capacity to the
        if adj != None:
            self.C = np.asarray(C)
        else:
            self.C = np.zeros((self.n, self.n))
            self.capacity = defaultdict(list)
            # Get capacity
            for val, edge in enumerate(edges):
                self.capacity[edge] = C[val]
                self.C[self.nodes.index(edge[0]), self.nodes.index(edge[1])] = C[val]
        
    def BFS(self,s, t, parent):
        """ Perform breath first search for capacity network """
        # Mark all the vertices as not visited
        visited = [False]*(self.n)
        # Create a queue for BFS
        queue = []
        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[self.nodes.index(s)] = True
        # Standard BFS Loop
        while queue:
            # Dequeue a vertex from queue and print it
            u = queue.pop(0)
            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.C[self.nodes.index(u)]):
                if visited[ind] == False and val > 0 :
                    queue.append(self.nodes[ind])
                    visited[ind] = True
                    parent[ind] = u
        # If we reached sink in BFS starting from source, then return
        # true, else false
        return True if visited[self.nodes.index(t)] else False

    # Ford Fulkeson algorithm
    def FordFulkerson(self, source, sink):
        """ Ford Fulkesson algorithm to find maximum flow """
		# This array is d by BFS and to store path
        parent = [-1]*(self.n)
        max_flow = 0 # There is no flow initially
		# Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent):
            # Find minimum residual capacity of the edges along the
			# path filled by BFS. Or we can say find the maximum flow
			# through the path found.
            path_flow = float("Inf")
            s = sink
            while(s != source):
                path_flow = min(path_flow, self.C[self.nodes.index(parent[self.nodes.index(s)])]
                                                 [self.nodes.index(s)])
                s = parent[self.nodes.index(s)]
            # Add path flow to overall flow
            max_flow += path_flow
            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while(v != source):
                u = parent[self.nodes.index(v)]
                self.C[self.nodes.index(u)][self.nodes.index(v)] -= path_flow
                self.C[self.nodes.index(v)][self.nodes.index(u)] += path_flow
                v = parent[self.nodes.index(v)]

        return max_flow
    
    def max_EDP(self, A, B):
        """ Return the max number of edge-disjoint paths from set A do B """
        nodes = self.nodes.copy()
        edges = self.edges.copy()
        # Insert "SSS" and "TTT"
        nodes.insert(0, "SSS")
        nodes.append("TTT")
        # Double the nodes other than SSS and TTT
        
        for a in A:
            edges.append((nodes[0], a))
        for b in B:
            edges.append((b, nodes[-1]))
        
        Exp = FlowGraph(nodes, edges, [1 for x in range(len(edges))])
        
        return Exp.FordFulkerson(self.nodes[0], self.nodes[-1])

    def max_VDP(self, A, B):
        """ Return the max number of vertex-disjoint paths from set A do B """
        # Define another FlowGraph with source to out-neighbors of A
        # Define another FlowGraph with
        # Set all capacities to 1
        #
        nodes = []
        edges = []
        An = []
        Bn = []
        for node in self.nodes:
            # Create two copy of nodes
            nodea = str(node) + "a"
            nodeb = str(node) + "b"
            # One node get the in-neighbors: node a
            for inn in self.inneighbors[node]:
                # Add an edge
                edges.append((str(inn)+"b", nodea))
            for out in self.outneighbors[node]:
                # Add an edge
                edges.append((nodeb, str(out)+"a"))
            # Add the edge connection two nodes
            edges.append((nodea, nodeb))
            # Add both nodes
            nodes.append(nodea)
            nodes.append(nodeb)
            if node in A:
                An.append(nodea)
            if node in B:
                Bn.append(nodeb)
                
        # Double the nodes
        # Insert "SSS" and "TTT"
        nodes.insert(0, "SSS")
        nodes.append("TTT")
        
        # Add edges from "SSS" to na
        for node in An:
            edges.append(("SSS", node))
        # Add edges from nb to "TTT"
        for node in Bn:
            edges.append((node, "TTT"))
        
        # Double the nodes other than SSS and TTT
        Exp = FlowGraph(nodes, edges, [1 for x in range(len(edges))])
        
        # Find the maximum flow of the extended graph 
        return Exp.FordFulkerson("SSS", "TTT")
        
# Symbolic functions
def DAG_create_sym_variables(A, labels=[], sep=None):
    """ Create the sym variables for a """
    n = len(A)
    if n > 10:
        sep = ','
    else:
        if sep is None:
            sep = ''
    m = len(labels)
    if m == 0:
        labels = list(range(n))
    Dict = {}
    As = [A]
    Allpaths = A.copy()
    for i in range(2, n):
        As.append(np.linalg.matrix_power(A, i))
        Allpaths += As[-1]

    pars = []
    Gs = sym.zeros(n, n)
    Ts = sym.zeros(n, n)

    for j in range(n):
        for i in range(n):
            Ts[j, i] = sym.Symbol('T_{'+str(labels[j])+sep+str(labels[i])+'}')
            if (Allpaths[j, i] == 0):
                Dict.update({Ts[j, i]: 0})
            if (j == i):
                Dict.update({Ts[j, i]: 1})
            if (j < i):
                Dict.update({Ts[j, i]: 0})
            if (j > i):
                if A[j, i] == 1:
                    Gs[j, i] = sym.Symbol('G_{'+str(labels[j])+sep+str(labels[i])+'}')
                    pars.append(Gs[j, i])

    return (Gs, Ts, Dict)

def DAG_solve_id(A, Gs, Ts, Dict):
    """ Get the dictionary from a structure """
    UnksG = Gs.copy()
    DictC = Dict.copy()
    UnksTs = set({})
    n = len(A)
    As = [A]
    Allpaths = A.copy()
    for i in range(2, n):
        As.append(np.linalg.matrix_power(A, i))
        Allpaths += As[-1]

    for j in range(n):
        for i in range(n):
            T = 1

def design_EMP_4_its(its):
    """ Design EMPs for induced trees list """
    B = set()
    C = set()
    BorC = set()
    for it in its:
        # Add all vertices to BorC
        BorC.update(it.nodes)
        # Check sources
        B.update(it.sources)
        # Check sinks
        C.update(it.sinks)
        # Remove B and C from BorC
        BorC = BorC.difference(B)
        BorC = BorC.difference(C)
        
    return B, C, BorC
        
def design_EMP_4_ppns(ppns):
    """ Desgin EMP for ppn list"""
    B = set()
    C = set()
    BorC = set()
    
    for ppn in ppns:
        # Add all nodes to BorC
        BorC.update(ppn.nodes)
        # Update B and C
        B.update(ppn.sources)
        C.update(ppn.sinks)
        BorC = BorC.difference(B)
        BorC = BorC.difference(C)
        # Get source
        source = ppn.sources[0]
        sink   = ppn.sinks[0]
        # Get all paths from source to sink
        paths = ppn.find_all_paths_i_to_j_(sink, source)
        # Sort the paths
        paths.sort(key=len)
        # Check the number of paths
        lofeachpath = [len(path)-1 for path in paths]
        # Check minimum value
        Min = np.min(lofeachpath)
        # Check if there is a direct edge from source to sink
        if Min == 1:
            for k, path in enumerate(paths):
                # All paths must obey the rule of one excited after one measured
                # Get number of nodes
                # Excite the first and measure the last one
                if lofeachpath[k] >= 2:
                    B.update(path[1:2])
                    C.update(path[-2:-1])
                    BorC = BorC.difference(B)
                    BorC = BorC.difference(C)
                    
        else:
            # Choose np - 1 paths, choose the largest one first
            for k, path in enumerate(paths[1:]):
                # All paths must obey the rule of one excited after one measured
                # Get number of nodes
                # Excite the first and measure the last one
                if lofeachpath[k] > 2:
                    B.update(path[1])
                    C.update(path[-2])
                    BorC = BorC.difference(B)
                    BorC = BorC.difference(C)
        
    return B, C, BorC

def check_equations(LHS, RHS, Known, Unk, InitSol):
    """ Checks whether it is possible to solve a list of symbolic equations """
    nlhs = len(LHS)
    nrhs = len(RHS)
    Kn = Known.copy()
    Uk = Unk.copy()
    Sol = dict()
    cond = False
    solidx = []
    eqns = [None]*nlhs
    j = 0
    while not cond:
        # If the solution has been found for this particular equation, try the next
        if j in solidx:
            j += 1
            if j >= nlhs:
                cond = True
                continue

        # Get the left hand side and right hand side
        lhs = sym.expand(LHS[j])
        rhs = sym.expand(RHS[j])
        # Get arguments from the left hand side
        argslhs = lhs.args
        symslhs = lhs.free_symbols
        # Check the known arguments
        for arg in argslhs:
            # Get symbols for args
            symarg = arg.free_symbols
            allargs = all([syms in Kn for syms in symarg])
            # Pass the argument to the other side
            if allargs:
                lhs = lhs - arg
                rhs = rhs - arg
        # Check the number of unknowns
        argsunk = lhs.free_symbols
        # Solved
        if len(argsunk) <= 1 and (j not in solidx):
            # Add to known
            Kn.update(argsunk)
            Sol.update({lhs: rhs})
            solidx.append(j)
            # Return to zero to check
            j = 0
            continue
        else:
            # Add to the updated lists
            eqns[j] = (lhs, rhs)
        # Update LHS and RHS
        LHS[j] = lhs
        RHS[j] = rhs
        # Test if the condition has changes
        if j >= nlhs:
            cond = True
        else:
            j += 1
    return (Sol, LHS, RHS, Kn, Uk)

def TjiR(j, i, G, T):
    """ Returns the inverse of T as a function of T considering the zeros"""
    Nkp = np.where(G[:, i])[0]
    S = 0
    for nkp in Nkp:
        S += T[j, nkp]*G[nkp, i]
    return S

def TjiL(j, i, G, T):
    """Returns the inverse of T as a function of T considering the zeros"""
    # Get the in-neighbors of j
    Nkm = np.where(G[j, :])[1]
    S = 0
    for nkm in Nkm:
        S += G[j, nkm]*T[nkm, i]
    return S

# def TjiLr(j, i, G, T):
#     """Returns the inverse of T as a function of T"""
#     if j == i:
#         return 1
#     if i > j:
#         return 0
#     else:
#         S = T[j,i]-G[j, i]
#         Nkm = np.where(G[j, :])[1]
#         for nkm in Nkm:
#             S += TjiLr(j, nkm, G, T)*T[nkm, i]
#         return S
    


def SjiR(j, i, G, T):
    """ Returns the inverse of T as a function of T considering the zeros"""
    if i > j:
        return 0
    if j == i:
        return 1
    else:
        S = 0
        for k in range(i+1, j+1):
            if G[j, k] != 0 or j == k:
                S += -SjiR(j, k, G, T)*T[k, i]
        return S


def SjiL(j, i, G, T):
    """Returns the inverse of T as a function of T considering the zeros"""
    if i > j:
        return 0
    if j == i:
        return 1
    else:
        S = 0
        for k in range(i, j):
            if G[k, i] != 0 or i == k:
                S += -T[j, k]*SjiL(k, i, G, T)
        return S

def DAG_algorithm(Gs, T, Initial):
    """ Returns the solution """
    IniSol = Initial.copy()
    # Get a graph
    Known = set()
    Unknown = set()
    n = Gs.shape[0]
    [Unknown.update({i}) for i in Gs if i != 0]
    [Unknown.update({i}) for i in T.subs(IniSol) if i != 0 and i != 1]
    # Get Adjacency Matrix
    A = np.array(Gs)
    A[A != 0] = 1
    # Get a graph
    G = digraph(n, A, 'adjacency')
    # Sources, Sinks, Dources, Dinks
    Sources = G.sources
    Sinks = G.sinks
    Dources = G.dources
    Dinks = G.dinks
    # Excited Nodes
    B = set()
    Baux = set()
    C = set()
    Caux = set()
    # Add Sources and Dources to B
    B.update(Sources)
    B.update(Dources)
    # Add Sinks and Dinks to C
    C.update(Sinks)
    C.update(Dinks)
    # Update known Ts from C B
    for c in C:
        for b in B:
            Known.update({T[c, b]})
    # Update Unknown
    Unknown = Unknown - Known
    # Check the equations for each column
    finished = False
    col = 0
    # All equations
    LHST = []
    RHST = []
    LHSG = []
    RHSG = []
    LHSAT = []
    RHSAT = []
    LHSAG = []
    RHSAG = []
    EQV = []
    for j in range(n):
        for i in range(n):
            # Check the network matrix
            if (j >= i):
                rs = Gs[j, i]
                ls = -SjiR(j, i, Gs, T)
                ls2 = -SjiL(j, i, Gs, T)
                if rs == 0 and (ls != -1 and ls != 0):
                    RHST.append(rs)
                    LHST.append(ls)
                    if (sym.simplify(ls - ls2) != 0):
                        RHST.append(rs)
                        LHST.append(ls2)
                if rs != 0 and (ls != -1 and ls != 0):
                    RHSG.append(rs)
                    LHSG.append(ls)
                    if (sym.simplify(ls - ls2) != 0):
                        RHSG.append(rs)
                        LHSG.append(ls2)

    # Check every equation
    while not finished:
        AddtoEMP = True
        lhs = LHST.copy()
        rhs = RHST.copy()
        sol, lhs, rhs, kn, unk = check_equations(lhs, rhs, Known, Unknown, IniSol)

        # Check if a new solution has been found
        for (key, value) in sol.items():
            if (key, value) not in IniSol.items():
                # Add the pair key value
                IniSol.update({key: value})
                # Add to the Known
                Known.update({key})
                # Remove it from unknown
                if key in Unknown:
                    Unknown.remove(key)
                AddtoEMP = False

        # Update Known and Unknown

        if AddtoEMP:
            solg, lhs, rhs, kn, unk = check_equations(LHSG, RHSG, Known, Unknown, IniSol)
            # If no solution is found add to the EMP
            if len(solg) == 0:
                # Start by adding to the emp
                # Check the minimum number to add to the EMPs
                EMP = 1
        else:
            # Check if the new solutions are ready
            solg, lhs, rhs, kn, unk = check_equations(LHSG, RHSG, Known, Unknown, IniSol)





        if len(Unknown) == 0:
            finished = True

    return 0

def findsourcesandsinks(G):
    """Returns the set of sources and sinks"""
    B = set()
    C = set()
    n = G.shape[0]
    for i in range(n):
        if sum(G[i, :]) == 0:
            B.add(i)
        if sum(G[:, i]) == 0:
            C.add(i)
    return B, C

def finddourcesanddinks(G):
    """Returns the set of dources and dinks"""
    B = set()
    C = set()
    n = G.shape[0]
    for i in range(n):
        # Find neighbors
        Nm = np.nonzero(np.array(G[i, :]).reshape((n, )))[0].tolist()
        Np = np.nonzero(np.array(G[:, i]).reshape((n, )))[0].tolist()
        if Nm == [] or Np == []:
            continue
        # get the matrix
        Gaux = G[np.ix_(Np, Nm)]
        lnp = len(Np)
        lnm = len(Nm)
        for k in range(lnp):
            if len(np.nonzero(np.array(Gaux[k:k+1, :]).reshape((lnm, )).tolist())[0]) == lnm:
                B.add(i)
                break
        for k in range(lnm):
            if len(np.nonzero(np.array(Gaux[:, k:k+1]).reshape((lnp, )).tolist())[0]) == lnp:
                C.add(i)
                break
    return B, C

def net_create_sym_variables(A, labels=[], sep=','):
    """ Create the sym variables for a """
    n = len(A)
    m = len(labels)
    if n != m and m > 0:
        raise ValueError("Labels must be an integer")
    if m == 0:
        labels = [str(x) for x in range(n)]

    Gs = sym.zeros(n, n)
    Ts = sym.zeros(n, n)

    # Get len of labels
    labels = [str(label) for label in labels]
    Lens = [len(label) for label in labels]

    for j in range(n):
        for i in range(n):
            if Lens[i] == 1 and Lens[j] == 1:
                Ts[j, i] = sym.Symbol('T_{'+labels[j]+labels[i]+'}')
                if A[j][i] == 1:
                    Gs[j, i] = sym.Symbol('G_{'+labels[j]+labels[i]+'}')

            else:
                Ts[j, i] = sym.Symbol('T_{'+labels[j]+sep+labels[i]+'}')
                if A[j, i] == 1:
                    Gs[j, i] = sym.Symbol('G_{'+labels[j]+sep+labels[i]+'}')


    return (Gs, Ts)

def generate_random_DAG(n):
    """Generate a random DAG with n nodes"""
    A = np.random.randint(0, 2, size=(n, n))
    return np.tril(A, k=-1)

def generate_random_net(n):
    """ Generate a random dynamic network with n nodes """
    # Generate a random adjacency matrix 
    A = np.random.randint(0, 2, size=(n, n))
    return np.tril(A, k=-1) + np.triu(A, k=1)

def find_EMPs_with_card(A, k, alg='Antoine'):
    """ 
         Find minimal EMPs with cardinality n + k
         Use it only with small network cardinalities
    """
    # Get all minimal EMPs with cardinality n + k
    n = len(A)
    tEMPs = choose_EMP_cardn(n, n+k)
    vEMPs = []
    iEMPs = []

    for emp in tEMPs:
        # Find the sources, and dources
        F, S = findsourcesandsinks(A)
        Df, Ds = finddourcesanddinks(A)
        # Nodes that need to be excited
        Exc = F.union(Df)
        Mea = S.union(Ds)
        # Check if the EMPs obey the restrictions
        if not Exc.issubset(set(emp[0])):
            iEMPs.append((emp[0], emp[1], Exc))
        elif not Mea.issubset(set(emp[1])):
            iEMPs.append((emp[0], emp[1], Mea))
        elif not set(range(n)).issubset(set(emp[0]).union(set(emp[1]))):
            iEMPs.append((emp[0], emp[1], set(emp[0]).union(set(emp[1]))))
        else:
            if alg == 'Antoine':
                out, i_edges, ni_edges = AntoineId(A, emp[0], emp[1], nsamples=3)
                if out:
                    vEMPs.append((emp[0], emp[1]))
                else:
                    iEMPs.append((emp[0], emp[1], ni_edges))
            else:
                return NotImplemented
    return vEMPs, iEMPs

def find_minimal_EMPS(A, alg='Antoine'):
    """ Find all minimal EMPs """
    # Get the number of nodes
    n = len(A)
    
    for k in range(n):
        # Find EMPs
        vEMPs, iEMPs = find_EMPs_with_card(A, k)
        # Check if there is at least one valid EMP
        if vEMPs != []:
            break
    
    return vEMPs, iEMPs

def find_EMP_cardk_nk(A, k, alg='Antoine'):
    """ Find all valid EMPs with cardinality k"""
    n = len(A)
    nemps = 0
    cont = 0
    EMPs = []
    # Find the sources, and dources
    F, S = findsourcesandsinks(A)
    Df, Ds = finddourcesanddinks(A)
    
    # All combinations 
    combs = combinations(list(range(2*n)), k)
    # Total combinations
    total_comb = comb(2*n, k)
    total25 = int(total_comb/4)
    total50 = int(total_comb/2)
    total75 = total25 + total50
    print(f"Total of {total_comb} EMPs to be tested")
    
    # Iterate over the combinations
    for emp in combs:
        cont += 1
        
        if cont == total25 or cont == total50 or cont == total75:
            print(f"Tested a total of {cont} EMPs")
            
        B = []
        C = []
        for x in emp:
            if x >= n:
                C.append(x-n)
            else:
                B.append(x)
        # Check if C is empty or B is empty
        if B != [] and C != []:
            # Check if all sources and dources are excited
            # Check if all sinks and dinks are excited
            if (F.issubset(B) and S.issubset(C)) and \
               (Df.issubset(B) and Ds.issubset(C)):
                # Check Identifiability of B and C
                # print(f"Test {cont}")
                out, i_edges, ni_edges = AntoineId(A, B, C, nsamples=3)
                # print(f"Testes {cont}")
                cont += 1
                if out:
                    # Print and Store
                    EMPs.append((B, C))
                    print(f"Valid EMP found: {(B, C)}")
                    nemps += 1

    return EMPs

def find_EMP_add_BC_card(A, bf, cf, nbk, nck, alg='Antoine'):
    """ """
    n = len(A)
    
    nbf = len(bf)
    ncf = len(cf)
    
    Exadd = nbk- nbf
    Msadd = nck- ncf
    
    # Remove from the lists 
    Ba = [b for b in range(n) if b not in bf]
    Ca = [c for c in range(n) if c not in cf]
    
    Choose = [Exadd, Msadd]
    Cfrom  = [Ba, Ca]
    
    # Get all combinations
    Combs = Comb_nk(Cfrom, Choose)
    
    Results = []
    
    # Start with the combinations
    for comb in Combs:
        B = bf.copy()
        C = cf.copy()
        # Add from bk
        B.append(comb[0])
        C.append(comb[1])
        # Test 
        out, i_edges, ni_edges = AntoineId(A, B, C, nsamples=3)
        
        Results.append((B, C, i_edges, ni_edges))
    
    return Results

def find_EMP_add_BC_card_aux(A, bf, cf, ba, ca, nbk, nck, alg='Antoine'):
    """ """
    n = len(A)
    
    nbf = len(bf)
    ncf = len(cf)
    
    Exadd = nbk- nbf
    Msadd = nck- ncf
    
    # Remove from the lists 
    # Ba = [b for b in range(n) if b not in bf]
    # Ca = [c for c in range(n) if c not in cf]
    
    Choose = [Exadd, Msadd]
    Cfrom  = [ba, ca]
    
    # Get all combinations
    Combs = Comb_nk(Cfrom, Choose)
    
    Results = []
    
    # Start with the combinations
    for comb in Combs:
        B = bf.copy()
        C = cf.copy()
        # Add from bk
        B += comb[0]
        C += comb[1]
        # Get unique elements
        B = list(set(B))
        C = list(set(C))
        # Test 
        out, i_edges, ni_edges = AntoineId(A, B, C, nsamples=3)
        
        Results.append((B, C, i_edges, ni_edges))
    
    return Results

def find_EMP_add_BC_card_aux_vdp_nip(A, bf, cf, ba, ca, nbk, nck, node, alg='Antoine'):
    """ """
    n = len(A)
    
    # Get the Ni
    Ax = np.asarray(A)
    
    Nip = np.argwhere(Ax[:, node] == 1)
    
    nbf = len(bf)
    ncf = len(cf)
    
    Exadd = nbk- nbf
    Msadd = nck- ncf
    
    # Remove from the lists 
    # Ba = [b for b in range(n) if b not in bf]
    # Ca = [c for c in range(n) if c not in cf]
    
    Choose = [Exadd, Msadd]
    Cfrom  = [ba, ca]
    
    # Get all combinations
    Combs = Comb_nk(Cfrom, Choose)
    
    Results = []
    
    # Start with the combinations
    for comb in Combs:
        B = bf.copy()
        C = cf.copy()
        # Add from bk
        B += comb[0]
        C += comb[1]
        # Get unique elements
        B = list(set(B))
        C = list(set(C))
        # Make a FlowGraph
        G = FlowGraph(list(range(n)), A, A, adj=True)
        # Take the number of vdps
        maxvdps = G.max_VDP(Nip, C)
        # Test 
        out, i_edges, ni_edges = AntoineId(A, B, C, nsamples=3)
        
        Results.append((B, C, i_edges, ni_edges, maxvdps))
    
    return Results
    
def find_EMP_combine_BC_card(A, bf, cf, bk, ck, k, alg='Antoine'):
    """ 
        Find all EMPs with cardinality B and C
        bf: fixed excitations
        cf: fixed measurements
        bk: to be added excitations
        ck: to be added measurements
    """
    n = len(A)
    
    nbk = len(bk)
    nck = len(ck)
    
    # Find the sources, and dources
    F, S = findsourcesandsinks(A)
    Df, Ds = finddourcesanddinks(A)
 
    # Check the desired k
    L = list(range(2*n))
    # Remove from the list the nbk elements
    for b in bk:
        if b in L:
            L.remove(b)
    # Remove from the list the ck elements
    for c in ck:
        if c + n in L:
            L.remove(c+n)
    
    k -= (nbk + nck)
    
    combs = combinations(L, k)
    
    nemps = 0
    EMPs = []
    EMPsni = []
    
    # Test
    for it, emp in enumerate(combs):
        # Make B and C
        B = bk
        C = ck
        for x in emp:
            if x >= n:
                C.append(x-n)
            else:
                B.append(x)
        # Test the EMP
        if B != [] and C != []:
            # Check if all sources and dources are excited
            # Check if all sinks and dinks are excited
            if (F.issubset(B) and S.issubset(C)) and \
               (Df.issubset(B) and Ds.issubset(C)):
                # Check Identifiability of B and C
                # print(f"Test {cont}")
                out, i_edges, ni_edges = AntoineId(A, B, C, nsamples=3)
                # print(f"Testes {cont}")
                if out:
                    # Print and Store
                    EMPs.append((B, C))
                    print(f"Valid EMP found: {(B, C)}")
                    nemps += 1
                else:
                    EMPsni.append((B, C, i_edges, ni_edges))

    return EMPs, EMPsni    


def find_EMPs_with_BC_card(A, bk, ck, k, alg='Antoine'):
    """ Find all EMPs with cardinality B and C"""
    n = len(A)
    
    nbk = len(bk)
    nck = len(ck)
    
    # Find the sources, and dources
    F, S = findsourcesandsinks(A)
    Df, Ds = finddourcesanddinks(A)
 
    
    # Check the desired k
    L = list(range(2*n))
    # Remove from the list the nbk elements
    for b in bk:
        if b in L:
            L.remove(b)
    # Remove from the list the ck elements
    for c in ck:
        if c + n in L:
            L.remove(c+n)
    
    k -= (nbk + nck)
    
    combs = combinations(L, k)
    
    nemps = 0
    EMPs = []
    EMPsni = []
    
    # Test
    for it, emp in enumerate(combs):
        # Make B and C
        B = bk
        C = ck
        for x in emp:
            if x >= n:
                C.append(x-n)
            else:
                B.append(x)
        # Test the EMP
        if B != [] and C != []:
            # Check if all sources and dources are excited
            # Check if all sinks and dinks are excited
            if (F.issubset(B) and S.issubset(C)) and \
               (Df.issubset(B) and Ds.issubset(C)):
                # Check Identifiability of B and C
                # print(f"Test {cont}")
                out, i_edges, ni_edges = AntoineId(A, B, C, nsamples=3)
                # print(f"Testes {cont}")
                if out:
                    # Print and Store
                    EMPs.append((B, C))
                    print(f"Valid EMP found: {(B, C)}")
                    nemps += 1
                else:
                    EMPsni.append((B, C, i_edges, ni_edges))

    return EMPs, EMPsni    
    
    

def find_EMPs_cardk_nk(A, k, alg='Antoine'):
    """ Find all valid EMPs with cardinality k"""
    n = len(A)
    nemps = 0
    cont = 0
    EMPs = []
    EMPsni = []
    # Find the sources, and dources
    F, S = findsourcesandsinks(A)
    Df, Ds = finddourcesanddinks(A)
    
    # All combinations 
    combs = combinations(list(range(2*n)), k)
    # Total combinations
    total_comb = comb(2*n, k)
    total25 = int(total_comb/4)
    total50 = int(total_comb/2)
    total75 = total25 + total50
    print(f"Total of {total_comb} EMPs to be tested")
    
    # Iterate over the combinations
    for emp in combs:
        cont += 1
        if cont % 100000 == 0:
            print(f"Tested a total of {cont} EMPs so far!")
        
        if cont == total25 or cont == total50 or cont == total75:
            print(f"Tested a total of {cont} EMPs")
            
        B = []
        C = []
        for x in emp:
            if x >= n:
                C.append(x-n)
            else:
                B.append(x)
        # Check if C is empty or B is empty
        if B != [] and C != []:
            # Check if all sources and dources are excited
            # Check if all sinks and dinks are excited
            if (F.issubset(B) and S.issubset(C)) and \
               (Df.issubset(B) and Ds.issubset(C)):
                # Check Identifiability of B and C
                # print(f"Test {cont}")
                out, i_edges, ni_edges = AntoineId(A, B, C, nsamples=3)
                # print(f"Testes {cont}")
                cont += 1
                if out:
                    # Print and Store
                    EMPs.append((B, C))
                    print(f"Valid EMP found: {(B, C)}")
                    nemps += 1
                else:
                    EMPsni.append((B, C, i_edges, ni_edges))

    return EMPs, EMPsni

    
    # Generate all combintations

def choose_EMP_cardn(n, k):
    """ Return all EMPs with cardinality k"""
    cEMPs = list(itertools.combinations(list(range(2*n)), k))
    EMPs = []
    for emp in cEMPs:
        B = []
        C = []
        for x in emp:
            if x >= n:
                C.append(x-n)
            else:
                B.append(x)
        # Check if C is empty or B is empty
        if B != [] and C != []:
            EMPs.append((B, C))
        
    return EMPs
            
def generate_all_combinations(L):
    """ Find all combinations of list L"""
    return itertools.chain(*map(lambda x: itertools.combinations(L, x), range(0, len(L)+1)))

def find_all_EMPs(A, alg='Antoine'):
    """ Find all combinations of EMPs """
    n = len(A)
    # Choose the algorithm
    if alg == 'Antoine':
        # Find the sources, and dources
        F, S = findsourcesandsinks(A)
        Df, Ds = finddourcesanddinks(A)
        # Nodes that need to be excited
        Exc = F.union(Df)
        Mea = S.union(Ds)
        # Generate all possible combinations
        Bs = list(generate_all_combinations(list(range(n))))
        Cs = list(generate_all_combinations(list(range(n))))
        # Initialze valid EMPs
        vEMPs = []
        iEMPs = []
        for B in Bs:
            for C in Cs:
                Alreadyseen = False
                # Check if a source or dource is not excited
                if not Exc.issubset(set(B)):
                    iEMPs.append((list(B), list(C), Exc))
                # Check if a sink or a dink is not measured
                elif not Mea.issubset(set(C)):
                    iEMPs.append((list(B), list(C), Mea))
                # Check if all node is either excited or measured
                elif not set(range(n)).issubset(set(B).union(set(C))):
                    iEMPs.append((list(B), list(C), set(B).union(set(C))))
                else:
                    # Check if there is some valid EMP which has been already tested
                    Bset = set(B)
                    Cset = set(C)
                    for emp in vEMPs:
                        if set.issubset(Bset, emp[0]) and set.issubset(Cset, emp[1]):
                            # Do nothing
                            Alreadyseen = True
                    # if any([] for emp in vEMPs])
                    # Test the if the EMP is valid
                    lB = list(B)
                    lC = list(C)

                    if Alreadyseen:
                        vEMPs.append((lB, lC))
                    else:
                        out, i_edges, ni_edges = AntoineId(A, lB, lC, nsamples=3)
                        if out:
                            vEMPs.append((lB, lC))
                        else:
                            iEMPs.append((lB, lC, ni_edges))
    else:
        return NotImplemented
    return vEMPs, iEMPs

def find_all_EMPs_full(A, full='excited', alg='Antoine'):
    """ Find """
    n = len(A)
    # Full excitation 
    if full == 'excited':
        # Full excitation
        B = list(range(n))
        # Generate all possible combinations
        Cs = generate_all_combinations(list(range(n)))
        # Valid EMPs
        vEMPs = []
        # Invalid EMPs
        ivEMPs = []
        # Check algorithms
        for C in Cs:
            if len(list(C)) == 0:
                continue
            if alg == 'Antoine':
                out, i_edges, ni_edges = AntoineId(A, B, list(C), nsamples=5)
                if out:
                    vEMPs.append((B, list(C)))
                else:
                    ivEMPs.append((B, list(C), ni_edges))
            else:
                return NotImplemented
    # Full measurement case
    if full == 'measured':
        # Full measurement
        C = list(range(n))
        # Generate all possible combinations
        Bs = generate_all_combinations(list(range(n)))
        # Valid EMPs
        vEMPs = []
        # Invalid EMPs
        ivEMPs = []
        # Check agorithms
        for B in Bs:
            if len(list(B)) == 0:
                continue
            if alg == 'Antoine':
                out, i_edges, ni_edges = AntoineId(A, list(B), C, nsamples=5)
                if out:
                    vEMPs.append((list(B), C))
                else:
                    ivEMPs.append((list(B), C, ni_edges))
            else:
                return NotImplemented
        
    if full != 'excited' and full != 'measured':
        raise ValueError('Choose either full = "excited" or full "Measured"')
    return (vEMPs, ivEMPs)

def AntoineId(A, B, C, nsamples=0, relax=False):
    """
        Antoine's algorithm for identifiability
        
        parameters:
            A: numpy.ndarray
               Adjacency matrix of the network
            B: list or numpy.ndarray
               List of excited nodes
            C: list or numpy.ndarray
               List of measured nodes
        
        returns:
            out: bools
                Returns True if the network is identifable, False otherwise
            edges: list of tuples
                A list of identifable edges
            n_edges: list of tuples
                A list of nonidentifiable edges
    """
    # Tolerance
    tol = 1e-9
    # Number of nodes
    L = len(A)
    b = np.zeros(L)
    c = np.zeros(L)
    b[B] = 1
    c[C] = 1
    B = np.diag(b)
    C = np.diag(c)
    Id = np.eye(L)
    delta = A.T.reshape((-1, ))
    # Logical version of delta
    deltaLogic = delta != 0
    edges_col, edges_row = np.nonzero(A.T)
    
    s = np.sum(delta);
    ni = np.zeros((s, ))
    out = False
    
    if nsamples > 0:
        nsamples -= 1
        symbolic = False
    else:
        symbolic = True
    
    for j in range(nsamples):
        if symbolic:
            A1 = sym.MatrixSymbol(L, L)
            A2 = sym.MatrixSymbol(L, L)
        else:
            A1 = np.random.rand(L, L)
            A2 = np.random.rand(L, L)
        
        G1 = A1*A
        G2 = A2*A
        
        T1 = np.linalg.inv(Id-G1)
        if relax == 1 or relax == 2:
            T2 = np.linalg.inv(Id-G2)
        # check relax
        if relax == 1: # Bilinear relaxation
            K = np.kron((T1 @ B).T, C @ T2) 
        elif relax == 2:
            K1 = np.kron((T1 @ B).T, C @ T2)
            if symbolic:
                K1[K1 != 0] = 1
                A3 = sym.MatrixSymbol(L**2, L**2)
                K = A3 * K1
            else:
                K1logic = np.abs(K1) > tol
                K = np.random.rand(L**2, L**2) * K1logic
        else:
            K = np.kron((T1 @ B).T, C @ T1)
        Kr = K[:, deltaLogic]
        r = np.linalg.matrix_rank(Kr)
        
        if r == s:
            out = True
        else:
            # Orthonormal basis for the null space of Kr
            N = null_space(Kr) 
            if symbolic:
                Nlogic = N != 0
            else:
                Nlogic = np.abs(N) > tol
            n = np.any(Nlogic, axis = 1)
            ni = np.logical_or(n, ni)
            
    if out:
        i_edges = list(zip(edges_row, edges_col))
        ni_edges = [None]
    else:
        i = np.logical_not(ni)
        i_row = edges_row[i]
        i_col = edges_col[i]
        
        ni_row = edges_row[ni]
        ni_col = edges_col[ni]
        
        i_edges  = list(zip(i_row, i_col))
        ni_edges = list(zip(ni_row, ni_col))
    
    # Returns
    return out, i_edges, ni_edges

def shift_list(L):
    """ Shift a list by one """
    n = len(L)
    last_el = L[-1]
    nL = [0] + L
    nL[0] = L[-1]
    nL.pop()
    return nL

def find_ness_excited_measured(EMPs):
    """ Find the nodes that need to be excited and measured in the EMPs """
    Bs, Cs = list(zip(*EMPs))
    B = set(set.intersection(*map(set, Bs)))
    C = set(set.intersection(*map(set, Cs)))
    return B, C

#========================================================================#

# Auxiliary functions
# Combinations functions

def combinations(list_get_comb, length_combination):
    """ Generator to get all the combinations of some length of the elements of a list.

    :param list_get_comb: List from which it is wanted to get the combination of its elements.
    :param length_combination: Length of the combinations of the elements of list_get_comb.
    :return: Generator with the combinations of this list.
    """

    # Generator to get the combinations of the indices of the list
    def get_indices_combinations(sub_list_indices, max_index):
        """ Generator that returns the combinations of the indices

        :param sub_list_indices: Sub-list from which to generate ALL the possible combinations.
        :param max_index: Maximum index.
        :return:
        """
        if len(sub_list_indices) == 1:  # Last index of the list of indices
            for index in range(sub_list_indices[0], max_index + 1):
                yield [index]
        elif all([sub_list_indices[-i - 1] == max_index - i for i in
                  range(len(sub_list_indices))]):  # The current sublist has reached the end
            yield sub_list_indices
        else:
            for comb in get_indices_combinations(sub_list_indices[1:],
                                                 max_index):  # Get all the possible combinations of the sublist sub_list_indices[1:]
                yield [sub_list_indices[0]] + comb
            # Advance one position and check all possible combinations
            new_sub_list = []
            new_sub_list.extend([sub_list_indices[0] + i + 1 for i in range(len(sub_list_indices))])
            for new_comb in get_indices_combinations(new_sub_list, max_index):
                yield new_comb  # Return all the possible combinations of the new list

    # Start the algorithm:
    sub_list_indices = list(range(length_combination))
    for list_indices in get_indices_combinations(sub_list_indices, len(list_get_comb) - 1):
        yield [list_get_comb[i] for i in list_indices]

def combinations_known(list_get_comb, list_fixed, length_combination):
    """ Generator to get all the combinations of some length of the elements of a list.

    :param list_get_comb: List from which it is wanted to get the combination of its elements.
    :param length_combination: Length of the combinations of the elements of list_get_comb.
    :return: Generator with the combinations of this list.
    """

    # Generator to get the combinations of the indices of the list
    def get_indices_combinations(sub_list_indices, max_index):
        """ Generator that returns the combinations of the indices

        :param sub_list_indices: Sub-list from which to generate ALL the possible combinations.
        :param max_index: Maximum index.
        :return:
        """
        if len(sub_list_indices) == 1:  # Last index of the list of indices
            for index in range(sub_list_indices[0], max_index + 1):
                yield [index]
        elif all([sub_list_indices[-i - 1] == max_index - i for i in
                  range(len(sub_list_indices))]):  # The current sublist has reached the end
            yield sub_list_indices
        else:
            for comb in get_indices_combinations(sub_list_indices[1:],
                                                 max_index):  # Get all the possible combinations of the sublist sub_list_indices[1:]
                yield [sub_list_indices[0]] + comb
            # Advance one position and check all possible combinations
            new_sub_list = []
            new_sub_list.extend([sub_list_indices[0] + i + 1 for i in range(len(sub_list_indices))])
            for new_comb in get_indices_combinations(new_sub_list, max_index):
                yield new_comb  # Return all the possible combinations of the new list

    # Start the algorithm:
    sub_list_indices = list(range(length_combination))
    for list_indices in get_indices_combinations(sub_list_indices, len(list_get_comb) - 1):
        yield [list_get_comb[i] for i in list_indices]

class Comb_nk:
    def __init__(self, a, ks):
        self.Z = [list(combinations(L, ks[i])) for i, L in enumerate(a)] # k combinations of each list inside a
        self.ns = [len(i)-1 for i in self.Z] # length of the each combination in Z
        self.p = len(a)                      # length of a
        self.pos = self.p*[0]                # I will keep track of which combination in each will be returned
        self.pos[-1] = -1                    # I will increase the combination index for each return
        
    def __iter__(self):
        return self
    
    def __next__(self):
        Z = self.Z
        p = self.p        
        self.pos = self.next_i()
        if isinstance(self.pos, GeneratorExit):
            self.pos = self.p*[0]
            self.pos[-1] = -1 
            raise StopIteration
        return [z[c] for z,c in zip(Z, self.pos)]      
            
    def next_i(self):
        '''
        This will iterate the next combination
        '''
        pos = self.pos
        ns = self.ns
        p = self.p
        for i in range(p-1, -1, -1):
            c = pos[i]
            n = ns[i]    
            if c == n:
                pos[i] = 0
            else:
                pos[i] += 1
                return pos
        return GeneratorExit('End of the Combinations')