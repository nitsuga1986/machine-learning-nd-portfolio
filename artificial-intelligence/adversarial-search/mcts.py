import random, math, copy, time
EXPLORATION_PARAM = math.sqrt(2.0)
#EXPLORATION_PARAM = 2
#MAX_ITER = 40
MAX_ITER = 80
NodeCounter = 0

# python run_match.py -f -r 500 -o RANDOM
# python run_match.py -r 500 -o RANDOM
# python run_match.py -f -r 500 -o GREEDY
# python run_match.py -r 500 -o GREEDY
# python run_match.py -f -r 500 -o MINIMAX
# python run_match.py -r 500 -o MINIMAX

class node_mcts():
    def __init__(self, state, parent=None):
        global NodeCounter
        self.id = NodeCounter
        self.visits = 1
        self.reward = 0
        self.state = state
        self.children = []
        self.children_actions = []
        self.parent = parent
        #print("-------- Create node [id: {}]".format(self.id))
        NodeCounter += 1

    def add_child(self, child_state, action):
        child = node_mcts(child_state, self)
        self.children.append(child)
        self.children_actions.append(action)
        #print("-------- Add child node [id: {}]: {}".format(self.id, child_state))

    def update(self, reward):
        self.reward += reward
        self.visits += 1
        #print("------------ Update [id: {}]: {} / {}".format(self.id, self.reward, reward))

    def fully_explored(self):
        #print("------------ Fully explored [id: {}]: {}".format(self.id, len(self.children_actions) == len(self.state.actions())))
        return len(self.children_actions) == len(self.state.actions())
    
class MCTS():
    """ MCTS 4 steps
    # 1. select() - select best child / unexplored child.
    # 2. expand() - expand towards the most promising moves
    # 3. simulate() - simulate to the end of the game, get reward win=1, loss=-1
    # 4. backpropagate() - update all nodes with reward, from leaf node all the way back to the root
    Reference: https://www.youtube.com/watch?v=UXW2yZndl7U
    """
    
    def __init__(self,state):
        #print("-------- Add root node")
        self.root_node = node_mcts(state)
        
    def start(self):
        if self.root_node.state.terminal_test():
            return random.choice(state.actions())
        for _ in range(MAX_ITER):
            child = self.select(self.root_node)
            if child is None:
                continue
            reward = self.simulate(child.state)
            self.backpropagate(child, reward)
            
        best_child = self.UCB1_select(self.root_node)
        best_idx = self.root_node.children.index(best_child)
        best_action = self.root_node.children_actions[best_idx]
        #print(NodeCounter)
        return best_action
        
    def select(self, node):
        """
        Select a leaf node.
        If not fully explored, return an unexplored child node.
        Otherwise, return the child node with the best score.
            param: node
            return: node
        """
        while not node.state.terminal_test():
            if not node.fully_explored():
                return self.expand(node)
            node = self.UCB1_select(node)
        return node
    
    def expand(self, node):
        """
        Expand the search tree.
        Create one (or more) child nodes and choose node from one of them. 
        Child nodes are any valid moves from the game position.
            param: node
            return: node (children)
        """
        for action in node.state.actions():
            if action not in node.children_actions:
                child_state = node.state.result(action)
                node.add_child(child_state, action)
                return node.children[-1]

    def simulate(self, state):
        """
        Randomly search the descendant of the state, and return the reward
            param: state
            return: +-1 (reward)
        """
        player_id = state.player()
        while not state.terminal_test():
            action = random.choice(state.actions())
            state = state.result(action)
        return -1 if state._has_liberties(player_id) else 1   # 1 for the winner, -1 for the loser

    def backpropagate(self, node, reward):
        """
        Backpropagation
        Use the result to update information in the nodes on the path.
            param: node, reward
            return: 
        """
        while node != None:
            node.update(reward)
            node = node.parent
            reward = -reward            
            
    def UCB1_select(self, node, C = EXPLORATION_PARAM):
        """
        Find the child node with the best UCB1 score.
            param: node
            return: node
        """
        best_children = []
        best_score = float("-inf")
        for child in node.children:
            exploit = child.reward / child.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / child.visits)
            UCB1_score = exploit + C * explore
            if UCB1_score == best_score:
                best_children.append(child)
            elif UCB1_score > best_score:
                best_children = [child]
                best_score = UCB1_score
        return random.choice(best_children)
