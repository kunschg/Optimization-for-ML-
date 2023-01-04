#STILL don't run properly, need to make it in a Jupyter Notebook

### We need to create various building blocks for our implementation

#1. VJP for all primitive operations 
#2. Node class
#3. Topological sort
#4. Forward pass
#5. Backward pass

import numpy as np 


# 1. VJP for all primitive operations 

def dot(x, W):
  return np.dot(W, x)

def dot_make_vjp(x, W):
  def vjp(u):
    return W.T.dot(u), np.outer(u, x)
  return vjp

dot.make_vjp = dot_make_vjp

def add(a, b):
  return a + b

def add_make_vjp(a, b):
  gprime = np.ones(len(a))
  def vjp(u):
    return u * gprime, u * gprime
  return vjp

add.make_vjp = add_make_vjp



#2. Node class

class Node(object):

    def __init__(self, value=None, func=None, parents=None, name="" ): # Value stored in the node.

        self.value = value
        # Function producing the node.
        self.func = func
        # Inputs to the function.
        self.parents = [] if parents is None else parents
        # Unique name of the node (for debugging and hashing). self.name = name
        # Gradient / Jacobian.
        self.grad = 0
        if not name:
            raise ValueError("Each node must have a unique name.")
    
    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return "Node(%s)" % self.name

def create_dag(x):
  x1 = Node(value=np.array([x[0]]), name="x1")
  x2 = Node(value=np.array([x[1]]), name="x2")
  x3 = Node(func=np.exp, parents=[x1], name="x3")
  x4 = Node(func=dot, parents=[x2, x3], name="x4")
  x5 = Node(func=add, parents=[x1, x4], name="x5")
  x6 = Node(func=np.sqrt, parents=[x5], name="x6")
  x7 = Node(func=dot, parents=[x4, x6], name="x7")
  return x7
  
#3. Topological sort

def dfs(node, visited):
    visited.add(node)
    for parent in node.parents:
        if not parent in visited:
        # Yield parent nodes first. 
            yield from dfs(parent, visited)
    # And current node later.
    yield node

def topological_sort(end_node):
    visited = set()
    sorted_nodes = []
    # All non-visited nodes reachable from end_node.
    for node in dfs(end_node, visited):
        sorted_nodes.append(node)
    return sorted_nodes


#4. Forward pass

def evaluate_dag(sorted_nodes):
    for node in sorted_nodes:
        if node.value is None:
            values = [p.value for p in node.parents]
            node.value = node.func(*values)
    return sorted_nodes[-1].value

#5. Backward pass

def backward_diff_dag(sorted_nodes): 
    value = evaluate_dag(sorted_nodes) 
    m = value.shape[0] # Output size
    
    # Initialize recursion.
    sorted_nodes[-1].grad = np.eye(m)
    for node_k in reversed(sorted_nodes): 
        if not node_k.parents:
        # We reached a node without parents.
            continue
        # Values of the parent nodes.
        values = [p.value for p in node_k.parents]
        # Iterate over outputs.
        for i in range(m):
            # A list of size len(values) containing the vjps. 
            vjps = node_k.func.make_vjp(*values)(node_k.grad[i])
            for node_j, vjp in zip(node_k.parents, vjps): 
                node_j.grad += vjp
    
    return sorted_nodes

x = [2,1]
breakpoint()
y = x = [2,1]