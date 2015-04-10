# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from __future__ import division
from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        try:
          foodDist = min([abs(x1-newPos[0])+abs(y1-newPos[1]) for x1, y1 in newFood.asList()])
        except:
          foodDist = 0

        try:
          ghostPositions = [x.getPosition() for x in newGhostStates]
          ghostDist = min([abs(x1-newPos[0])+abs(y1-newPos[1]) for x1, y1 in ghostPositions])
        except:
          ghostDist = 0
        return successorGameState.getScore() - foodDist + ghostDist

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """


        actions = []

        def max_value(game_state, agent_num, depth):
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)

            v = max([(value(game_state.generateSuccessor(agent_num, action), agent_num + 1, depth),
                      action) for action in legal_moves], key=lambda x: x[0])
            actions.append(v[1])
            return v[0]

        def min_value(game_state, agent_num, depth):
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)
            v = min([value(game_state.generateSuccessor(agent_num, action), agent_num + 1, depth)
                     for action in legal_moves])
            return v

        def value(game_state, agent_num=0, depth=1):
            num_agents = game_state.getNumAgents()
            if agent_num >= num_agents:
                if game_state.isLose() or game_state.isWin():
                    return self.evaluationFunction(game_state)
                if self.depth > depth:
                    return value(game_state, 0, depth+1)
                else:
                    return self.evaluationFunction(game_state)

            if agent_num == 0:
                return max_value(game_state, agent_num, depth)
            else:
                return min_value(game_state, agent_num, depth)

        value(gameState)
        return actions[-1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        actions = []

        def max_value(game_state, agent_num, depth, alpha, beta):
            v = (-sys.maxint, None)
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)
            for action in legal_moves:
                successor_state = game_state.generateSuccessor(agent_num, action)
                v = max(v, (value(successor_state, agent_num+1, depth, alpha, beta), action), key=lambda x: x[0])
                if v[0] > beta:
                    actions.append(v[1])
                    return v[0]
                alpha = max(alpha, v[0])
            actions.append(v[1])
            return v[0]

        def min_value(game_state, agent_num, depth, alpha, beta):
            v = sys.maxint
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)
            for action in legal_moves:
                successor_state = game_state.generateSuccessor(agent_num, action)
                v = min(v, value(successor_state, agent_num+1, depth, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def value(game_state, agent_num=0, depth=1, alpha=-sys.maxint, beta=sys.maxint):
            num_agents = game_state.getNumAgents()
            if agent_num >= num_agents:
                if game_state.isLose() or game_state.isWin():
                    return self.evaluationFunction(game_state)
                if self.depth > depth:
                    return value(game_state, 0, depth+1, alpha, beta)
                else:
                    return self.evaluationFunction(game_state)

            if agent_num == 0:
                return max_value(game_state, agent_num, depth, alpha, beta)
            else:
                return min_value(game_state, agent_num, depth, alpha, beta)

        value(gameState)
        return actions[-1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """


        actions = []

        def max_value(game_state, agent_num, depth):
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)

            v = max([(value(game_state.generateSuccessor(agent_num, action), agent_num + 1, depth),
                      action) for action in legal_moves], key=lambda x: x[0])
            actions.append(v[1])
            return v[0]

        def min_value(game_state, agent_num, depth):
            legal_moves = game_state.getLegalActions(agent_num)
            if not legal_moves:
                return self.evaluationFunction(game_state)
            v = sum([value(game_state.generateSuccessor(agent_num, action), agent_num + 1, depth)
                     for action in legal_moves])/len(legal_moves)
            return v

        def value(game_state, agent_num=0, depth=1):
            num_agents = game_state.getNumAgents()
            if agent_num >= num_agents:
                if game_state.isLose() or game_state.isWin():
                    return self.evaluationFunction(game_state)
                if self.depth > depth:
                    return value(game_state, 0, depth+1)
                else:
                    return self.evaluationFunction(game_state)

            if agent_num == 0:
                return max_value(game_state, agent_num, depth)
            else:
                return min_value(game_state, agent_num, depth)

        value(gameState)
        return actions[-1]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    import math
    x, y = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    #print capsules

    newGhostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    try:
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
        ghostDist = min([abs(x1-x)+abs(y1-y) for x1, y1 in ghostPositions])
    except:
        ghostDist = 0

    if any(scaredTimes):
        ghostDist = sys.maxint

    try:
        foodDist = min([abs(x1-x)+abs(y1-y) for x1, y1 in foodGrid])
    except:
        foodDist = 0

    return -foodDist + math.log(ghostDist+1) + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

