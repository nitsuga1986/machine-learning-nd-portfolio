import random, math, copy, time
EXPLORATION_PARAM = math.sqrt(2.0)
MAX_ITER = 100
NodeCounter = 0


class ALPHA_BETA():
    def __init__(self, state, depth = 3):
        self.state = state
        self.player_id = state.player()
        self.deph = depth
        
    def start(self):
        best_action = self.alpha_beta_search(self.deph)
        #for depth in range(1, self.deph + 1):
        #    best_action = self.alpha_beta_search(self.deph)
        return best_action
    
    def alpha_beta_search(self, depth=3):
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_action = None
        for action in self.state.actions():
            value = self.min_value(self.state.result(action), alpha, beta, depth-1)
            alpha = max(alpha, value)
            if value >= best_score:
                best_score = value
                best_action = action
        return best_action

    def min_value(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)  # +inf:won | -inf:lose |  0: otherwise
        if depth <= 0: return self.score(state)                         # score
        value = float("inf")
        for action in state.actions():
            value = min(value, self.max_value(state.result(action), alpha, beta, depth-1))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def max_value(self, state, alpha, beta, depth):
        if state.terminal_test(): return state.utility(self.player_id)  # +inf:won | -inf:lose |  0: otherwise
        if depth <= 0: return self.score(state)                         # score
        value = float("-inf")
        for action in state.actions():
            value = max(value, self.min_value(state.result(action), alpha, beta, depth-1))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value
    
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
        
        
        
        