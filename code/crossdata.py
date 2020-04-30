"""
Read node_dict and edge_file 
Construct a heterogeneous graph
Add user mode version
20180923
Author: Liu Yang
"""
import numpy as np
from collections import defaultdict



class CrossData(object):
    def __init__(self, node_path, edge_path):
        self.node_path = node_path
        self.edge_path = edge_path
        self.node_type_list = ['t', 'w', 'l', 'u']
        self.node_dict, self.node_id2id, self.node_type = self.read_nodes()
        self.node_num = len(self.node_dict)        
        self.edge_num, self.linked_nodes, self.et2net = self.read_edges()
        self.edge_type = self.et2net.keys()
        # self.node_degree = self.get_degree()
        self.edges = self.get_edges()
        self.wd2id = self.get_word_id()


    def read_nodes(self):
        """
        Read node_dict file.
        Input format:
            node_id node_type intype_id value
            seperated by '\x01'
        Output format:
            node_id2nt is a dictionary whose key is the node id and
                value is a list [node_type, intype_id, value]
            node_type is a dictionary whose key is the node type and 
                value is a node_id list of this type
        """
        with open(self.node_path, "r") as f:
            node_lines = f.readlines()
        node_id2nt = {}
        node_id2id = {}
        node_type = {}
        for t in self.node_type_list:
            node_type[t] = []
            node_id2id[t] = {}
        for line in node_lines:
            line_lst = line.split('\x01')
            node_id = int(line_lst[0])
            node_id2nt[node_id] = [line_lst[1], line_lst[2], line_lst[3]]
            node_id2id[line_lst[1]][int(line_lst[2])] = node_id
            node_type[line_lst[1]].append(node_id)
        return node_id2nt, node_id2id, node_type


    def read_edges(self):
        """
        Read edge file.
        Input format:
            edge_type start_node_id end_node_id weight
            seperated by ' '
        Output format:
            line_num is the num of edges
            linked_nodes is a dictionary
                the first key is the node id
                the second key is the end_node_type
                value is a end_node list 
            non_linked_nodes is a dictionary
                the first key is the node id
                the second key is the end_node_type
                value is a non_linked_node list 
            et2net is a dictionary
                the first key is the edge type
                the second key is a 2-tuple (start_node, end_node)
                value is the weight
        """
        with open(self.edge_path, "r") as f:
            edge_lines = f.readlines()

        n_node = self.node_num
        linked_nodes = {}
        for j in range(n_node):
            linked_nodes[j] = {} # Initialization empty list for every node
            for t in self.node_type_list:
                linked_nodes[j][t] = []

        line_num = 0
        et2net = defaultdict(lambda : defaultdict(float))                
        for line in edge_lines:
            line_num = line_num + 1
            line_lst = line.split()
            linked_nodes[int(line_lst[1])][line_lst[0][1]].append(int(line_lst[2]))
            et2net[line_lst[0]][(int(line_lst[1]),int(line_lst[2]))] = float(line_lst[3])
        return line_num, linked_nodes, et2net


    def get_degree(self):
        """
        Calculate the node degree limited to edge type
        Output:
            u_degree is a dictionary
                the first key is the edge type
                the second key is the start node id
                the value is the certain edge type degree(sum of weights)
        """
        u_degree = {}
        for edge_tp in self.edge_type:
            u_degree[edge_tp] = {}
            node_list = self.node_type[edge_tp[0]]
            for u in node_list:
                u_degree[edge_tp][u] = 0.0
                for v in self.linked_nodes[u][edge_tp[1]]:
                    u_degree[edge_tp][u] += self.et2net[edge_tp][(u,v)]
        return u_degree


    def get_edges(self):
        edges = {}
        for et in self.et2net.keys():
            edges[et] = self.et2net[et].keys()
        return edges


    def get_word_id(self):
        """
        wd2id is a dictionary whose key is the word and value is its global id 
        """
        wd2id = {}
        for w_id in self.node_type['w']:
            word = self.node_dict[w_id][2]
            wd2id[word] = w_id
        return wd2id
