
import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import shuffle
from ast import literal_eval



class DotDict(dict):
    """Wrapper around in-built dict class to access members through the dot operation.
    """

    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class MSTReader(object):
    """Iterator that reads MST dataset files and yields mini-batches."""

    def __init__(self, num_nodes, num_neighbors, batch_size, filepath):
        """
        Args:
            num_nodes: Number of nodes in MST
            num_neighbors: Number of neighbors to consider for each node in graph
            batch_size: Batch size
            filepath: Path to dataset file (.txt file)
        """
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.filepath = filepath
        self.filedata = shuffle(open(filepath, "r").readlines())  # Always shuffle upon reading data
        self.max_iter = (len(self.filedata) // batch_size)

    def __iter__(self):
        for batch in range(self.max_iter):
            start_idx = batch * self.batch_size
            end_idx = (batch + 1) * self.batch_size
            yield self.process_batch(self.filedata[start_idx:end_idx])

    def process_batch(self, lines):
        """Helper function to convert raw lines into a mini-batch as a DotDict.
        """
        batch_edges = []
        batch_edges_values = []
        batch_edges_target = []  # Binary classification targets (0/1)
        batch_nodes = []
        batch_nodes_target = []  # Multi-class classification targets (`num_nodes` classes)
        batch_nodes_coord = []
        batch_mst_edges = []
        batch_mst_len = []

        for line_num, line in enumerate(lines):
            line = line.split(" ")  # Split into list
            
            # Compute signal on nodes
            nodes = np.ones(self.num_nodes)  # All 1s for TSP...
            
            # Convert node coordinates to required format
            nodes_coord = []
            for idx in range(0, 2 * self.num_nodes, 2):
                nodes_coord.append([float(line[idx]), float(line[idx + 1])])
            
            # Compute distance matrix
            W_val = squareform(pdist(nodes_coord, metric='euclidean'))
            
            # Compute adjacency matrix
            if self.num_neighbors == -1:
                W = np.ones((self.num_nodes, self.num_nodes))  # Graph is fully connected
            else:
                W = np.zeros((self.num_nodes, self.num_nodes))
                # Determine k-nearest neighbors for each node
                knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, self.num_neighbors::-1]
                # Make connections 
                for idx in range(self.num_nodes):
                    W[idx][knns[idx]] = 1
            np.fill_diagonal(W, 2)  # Special token for self-connections
            
            # Convert tour nodes to required format
            # Don't add final connection for tour/cycle
            _edges = [edge for edge in line[line.index('output') + 1:]]
            mst_edges = [literal_eval(' '.join(_edges[i:i+2])) for i in range(0, len(_edges),2)] # gets the edges from line --> equivalent to tour_nodes for TSP
            
            # Compute node and edge representation of mst + mst_len
            mst_len = 0 # length of mst
            nodes_target = np.zeros(self.num_nodes)
            edges_target = np.zeros((self.num_nodes, self.num_nodes))
            for idx in range(len(mst_edges)):
                i = mst_edges[idx][0]
                j = mst_edges[idx][1]
                nodes_target[i] = idx  # node targets: ordering of nodes in tour
                edges_target[i][j] = 1
                edges_target[j][i] = 1
                mst_len += W_val[i][j]
            
            # Concatenate the data
            batch_edges.append(W)
            batch_edges_values.append(W_val)
            batch_edges_target.append(edges_target)
            batch_nodes.append(nodes)
            batch_nodes_target.append(nodes_target)
            batch_nodes_coord.append(nodes_coord)
            batch_mst_edges.append(mst_edges)
            batch_mst_len.append(mst_len)
        
        # From list to tensors as a DotDict
        batch = DotDict()
        batch.edges = np.stack(batch_edges, axis=0)
        batch.edges_values = np.stack(batch_edges_values, axis=0)
        batch.edges_target = np.stack(batch_edges_target, axis=0)
        batch.nodes = np.stack(batch_nodes, axis=0)
        batch.nodes_target = np.stack(batch_nodes_target, axis=0)
        batch.nodes_coord = np.stack(batch_nodes_coord, axis=0)
        batch.mst_edges = np.stack(batch_mst_edges, axis=0)
        batch.mst_len = np.stack(batch_mst_len, axis=0)
        return batch

# TEST:
if __name__ == "__main__":
    dataset = MSTReader(num_nodes = 10, num_neighbors = 5, batch_size = 1, filepath = '../data/mst10_networkxMST.txt')
    batch = next(iter(dataset))
    print(batch)
