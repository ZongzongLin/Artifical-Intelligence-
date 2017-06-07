import numpy as np
import copy
import pygame
from pygame.locals import*
from sys import exit
import time
import math
import random


class TreeNode:

    def __init__(self, parentnode, state, parene_action, player):
        self.action = parene_action
        self.parent = parentnode
        self.wins = 0.0
        self.visits = 0.0
        self.unreach = feasible_actions(state, player)
        self.children = []
        self.activeplayer = player


    def addchild(self, state, parent_action, child_player):
        node = TreeNode(self, state, parent_action, child_player)
        self.unreach.remove(parent_action)
        self.children.append(node)
        return node

    def selectchild(self):
        bestValue = -1000000
        for i in range(len(self.children)):
            child = self.children[i]
            uctValue = child.wins/child.visits + \
                math.sqrt(2*math.log(self.visits)/child.visits)
            if uctValue > bestValue:
                selected = child
                bestValue = uctValue
        return selected

    def update(self, result):
        self.visits += 1.0
        self.wins +=int(int(result) == int(self.activeplayer))
        if not self.parent is None:
            self.parent.update(result)

    def mostVisitedchild(self):
        mostVisited = self.children[0]
        for i in range(len(self.children)):
            if self.children[i].visits > mostVisited.visits:
                mostVisited = self.children[i]
        return mostVisited


def UCT_MCTS(state, max_Iteration, max_time, turn,pos_weight):
    blocksize = 1000
    node_visited = 0.0
    root = TreeNode(None, state, None, turn)
    current_time = time.time()
    end = False
    for k in range(max_Iteration):
        if end:
            break
        for s in range(blocksize):
            if time.time() > max_time + current_time:
                end = True
                print time.time()-current_time
                break
            node = root
            temp_state = state.copy()
            # selection
            while len(node.unreach) == 0 and len(node.children) > 0:
                node = node.selectchild()
                temp_state = update_state(temp_state, node.action[
                                          0], node.action[1], node.activeplayer)
            # expansion
            if len(node.unreach) > 0:
                action = random.choice(node.unreach)
                temp_state = update_state(temp_state, action[0], action[
                                          1], node.activeplayer)
                node = node.addchild(temp_state, action, -1*node.activeplayer)
            # simulation
            now_turn=node.activeplayer
            actions = feasible_actions(temp_state, now_turn)
            while True:
                if len(actions)>0:
                    #action = random.choice(actions)
                    action=chose_epsilon_pos_weight(actions,pos_weight)
                    temp_state = update_state(temp_state, action[0], action[1], now_turn)
                    node_visited += 1
                    now_turn = -1*now_turn
                    actions = feasible_actions(temp_state, now_turn)
                else:
                    now_turn=-1*now_turn
                    actions = feasible_actions(temp_state, now_turn)
                    if len(actions)==0:
                        break
            # backpropagation
            result = game_winner(temp_state)
            while(node):
                node.update(result)
                node = node.parent
    print node_visited
    return root.mostVisitedchild().action

def chose_epsilon_pos_weight(actions,pos_weight):
    pos_pair=[pos_weight[action[0]][action[1]] for action in actions]
    sum_pair=float(sum(pos_pair))
    pair=map(lambda x:x/sum_pair,pos_pair)
    pairs=zip(pair,actions)
    q=random.random()
    cumulate=0.0
    for prob,action in pairs:
        if q<cumulate:
            cumulate+=prob
        else:
            return action



def game_not_end(temp_state):
    if feasible_actions(temp_state,1) or feasible_actions(temp_state,-1):
        return True
    else:
        return False


def game_winner(state):
    black = (state == -1).sum()
    white = (state == 1).sum()
    if  black<white:
        return 1
    elif white>black:
        return -1
    else:
        return 0.5


def feasible_actions(state, turn):
    feasible_actions = []
    for x in range(8):
        for y in range(8):
            temp_state = copy.deepcopy(state)
            if state[x][y] == 0 and quick_check_state(temp_state, x, y, turn) == True:
                feasible_actions.append((x, y))
    return feasible_actions


def quick_check_state(state, x, y, turn):
    changer = []
    for i in range(y + 1, 8, 1):
        if state[x][i] == -1 * turn:
            changer.append(i)
            if i == 7:
                changer = []
                break
        elif state[x][i] == turn:
            break
        else:
            changer = []
            break
    if changer != []:
        return True
        # left
    changel = []
    for i in range(y - 1, -1, -1):
        if state[x][i] == -1 * turn:
            changel.append(i)
            if i == 0:
                changel = []
                break
        elif state[x][i] == turn:
            break
        else:
            changel = []
            break
    if changel != []:
        return True

        # down
    changed = []
    for i in range(x + 1, 8, 1):
        if state[i][y] == -1 * turn:
            changed.append(i)
            if i == 7:
                changed = []
                break
        elif state[i][y] == turn:
            break
        else:
            changed = []
            break
    if changed != []:
        return True

        # up
    changeu = []
    for i in range(x - 1, -1, -1):
        if state[i][y] == -1 * turn:
            changeu.append(i)
            if i == 0:
                changeu = []
                break
        elif state[i][y] == turn:
            break
        else:
            changeu = []
            break
    if changeu != []:
        return True

        # lu
    changelu = []
    for i in zip(range(x - 1, -1, -1), range(y - 1, -1, -1)):
        if state[i[0]][i[1]] == -1 * turn:
            changelu.append(i)
            if i[0] == 0 or i[1] == 0:
                changelu = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changelu = []
            break
    if changelu != []:
        return True
        # ld
    changeld = []
    for i in zip(range(x + 1, 8, 1), range(y - 1, -1, -1)):
        if state[i[0]][i[1]] == -1 * turn:
            changeld.append(i)
            if i[0] == 7 or i[1] == 0:
                changeld = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changeld = []
            break
    if changeld != []:
        return True
        # ru
    changeru = []
    for i in zip(range(x - 1, -1, -1), range(y + 1, 8, 1)):
        if state[i[0]][i[1]] == -1 * turn:
            changeru.append(i)
            if i[0] == 0 or i[1] == 7:
                changeru = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changeru = []
            break
    if changeru != []:
        return True
        # rd
    changerd = []
    for i in zip(range(x + 1, 8, 1), range(y + 1, 8, 1)):
        if state[i[0]][i[1]] == -1 * turn:
            changerd.append(i)
            if i[0] == 7 or i[1] == 7:
                changerd = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changerd = []
            break
    if changerd != []:
        return True
    return False


def update_state(state, x, y, turn):
    state[x][y] = turn
    # right
    changer = []
    for i in range(y + 1, 8, 1):
        if state[x][i] == -1 * turn:
            changer.append(i)
            if i == 7:
                changer = []
                break
        elif state[x][i] == turn:
            break
        else:
            changer = []
            break
    for i in changer:
        state[x][i] = turn
        # left
    changel = []
    for i in range(y - 1, -1, -1):
        if state[x][i] == -1 * turn:
            changel.append(i)
            if i == 0:
                changel = []
                break
        elif state[x][i] == turn:
            break
        else:
            changel = []
            break
    for i in changel:
        state[x][i] = turn
        # down
    changed = []
    for i in range(x + 1, 8, 1):
        if state[i][y] == -1 * turn:
            changed.append(i)
            if i == 7:
                changed = []
                break
        elif state[i][y] == turn:
            break
        else:
            changed = []
            break
    for i in changed:
        state[i][y] = turn
        # up
    changeu = []
    for i in range(x - 1, -1, -1):
        if state[i][y] == -1 * turn:
            changeu.append(i)
            if i == 0:
                changeu = []
                break
        elif state[i][y] == turn:
            break
        else:
            changeu = []
            break
    for i in changeu:
        state[i][y] = turn
        # lu
    changelu = []
    for i in zip(range(x - 1, -1, -1), range(y - 1, -1, -1)):
        if state[i[0]][i[1]] == -1 * turn:
            changelu.append(i)
            if i[0] == 0 or i[1] == 0:
                changelu = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changelu = []
            break
    for i in changelu:
        state[i[0]][i[1]] = turn
    # ld
    changeld = []
    for i in zip(range(x + 1, 8, 1), range(y - 1, -1, -1)):
        if state[i[0]][i[1]] == -1 * turn:
            changeld.append(i)
            if i[0] == 7 or i[1] == 0:
                changeld = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changeld = []
            break
    for i in changeld:
        state[i[0]][i[1]] = turn
    # ru
    changeru = []
    for i in zip(range(x - 1, -1, -1), range(y + 1, 8, 1)):
        if state[i[0]][i[1]] == -1 * turn:
            changeru.append(i)
            if i[0] == 0 or i[1] == 7:
                changeru = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changeru = []
            break
    for i in changeru:
        state[i[0]][i[1]] = turn
    # rd
    changerd = []
    for i in zip(range(x + 1, 8, 1), range(y + 1, 8, 1)):
        if state[i[0]][i[1]] == -1 * turn:
            changerd.append(i)
            if i[0] == 7 or i[1] == 7:
                changerd = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changerd = []
            break
    for i in changerd:
        state[i[0]][i[1]] = turn
    return state


class reversi(object):

    def __init__(self):
        self.turn = -1
        self.state = np.zeros([8,8],int)
        self.state[3][3] = 1
        self.state[3][4] = -1
        self.state[4][3] = -1
        self.state[4][4] = 1

    def move(self, x, y):
        return self.update_state(x, y)

    def feasible_move_show(self):
        feasible_move = []
        for x in range(8):
            for y in range(8):
                temp_state = self.state.copy()
                if self.state[x][y] == 0 and type(self.complete_update_state(temp_state, x, y, self.turn)) == type(np.zeros(1)):
                    feasible_move.append((x, y))
        return feasible_move

    def update_state(self, x, y):
        if (x, y) not in self.feasible_move_show():
            return False
        self.state[x][y] = self.turn  # upated the change
        self.state = self.complete_update_state(self.state, x, y, self.turn)
        return True

    def complete_update_state(self, state, x, y, turn):
        # right
        changer = []
        for i in range(y+1, 8, 1):
            if state[x][i] == -1*turn:
                changer.append(i)
                if i == 7:
                    changer = []
                    break
            elif state[x][i] == turn:
                break
            else:
                changer = []
                break
        for i in changer:
            state[x][i] = turn
        # left
        changel = []
        for i in range(y-1, -1, -1):
            if state[x][i] == -1*turn:
                changel.append(i)
                if i == 0:
                    changel = []
                    break
            elif state[x][i] == turn:
                break
            else:
                changel = []
                break
        for i in changel:
            state[x][i] = turn
        # down
        changed = []
        for i in range(x+1, 8, 1):
            if state[i][y] == -1*turn:
                changed.append(i)
                if i == 7:
                    changed = []
                    break
            elif state[i][y] == turn:
                break
            else:
                changed = []
                break
        for i in changed:
            state[i][y] = turn
        # up
        changeu = []
        for i in range(x-1, -1, -1):
            if state[i][y] == -1*turn:
                changeu.append(i)
                if i == 0:
                    changeu = []
                    break
            elif state[i][y] == turn:
                break
            else:
                changeu = []
                break
        for i in changeu:
            state[i][y] = turn
        # lu
        changelu = []
        for i in zip(range(x-1, -1, -1), range(y-1, -1, -1)):
            if state[i[0]][i[1]] == -1*turn:
                changelu.append(i)
                if i[0] == 0 or i[1] == 0:
                    changelu = []
                    break
            elif state[i[0]][i[1]] == turn:
                break
            else:
                changelu = []
                break
        for i in changelu:
            state[i[0]][i[1]] = turn
        # ld
        changeld = []
        for i in zip(range(x+1, 8, 1), range(y-1, -1, -1)):
            if state[i[0]][i[1]] == -1*turn:
                changeld.append(i)
                if i[0] == 7 or i[1] == 0:
                    changeld = []
                    break
            elif state[i[0]][i[1]] == turn:
                break
            else:
                changeld = []
                break
        for i in changeld:
            state[i[0]][i[1]] = turn
        # ru
        changeru = []
        for i in zip(range(x-1, -1, -1), range(y+1, 8, 1)):
            if state[i[0]][i[1]] == -1*turn:
                changeru.append(i)
                if i[0] == 0 or i[1] == 7:
                    changeru = []
                    break
            elif state[i[0]][i[1]] == turn:
                break
            else:
                changeru = []
                break
        for i in changeru:
            state[i[0]][i[1]] = turn
        # rd
        changerd = []
        for i in zip(range(x+1, 8, 1), range(y+1, 8, 1)):
            if state[i[0]][i[1]] == -1*turn:
                changerd.append(i)
                if i[0] == 7 or i[1] == 7:
                    changerd = []
                    break
            elif state[i[0]][i[1]] == turn:
                break
            else:
                changerd = []
                break
        for i in changerd:
            state[i[0]][i[1]] = turn

        if len(changel+changer+changed+changeu+changeld+changelu+changerd+changeru) == 0:
            return False
        else:
            return state

    def winorlose(self):
        Black = (state == -1).sum()
        White = (state == 1).sum()
        if  Black>White:
            text_surface = my_font.render("Black win", True, (0, 0, 255))
            screen.blit(text_surface, (120, 150))
            pygame.display.update()
            print 'Black %d , White %d'%(Black,White)
            time.sleep(10)
            print 'Black win'
        elif Black<White:
            text_surface = my_font.render("White win", True, (0, 0, 255))
            screen.blit(text_surface, (120, 150))
            pygame.display.update()
            print 'Black %d , White %d' %(Black,White)
            time.sleep(10)
            print 'White win'
        elif Black==White:
            text_surface = my_font.render("Double win", True, (0, 0, 255))
            screen.blit(text_surface, (120, 150))
            pygame.display.update()
            print 'Black %d , White %d' %(Black,White)
            time.sleep(10)
            print 'Double win'

    def plot_graph(self):
        screen.blit(background, (0, 0))
        for i in range(8):
            for j in range(8):
                if self.state[i][j] == 1:
                    screen.blit(white, (simple2complex(j, i)))
                elif self.state[i][j] == -1:
                    screen.blit(black, (simple2complex(j, i)))
        pygame.display.update()

def quick_policy(flag):
    if len(flag) == 1:
        return flag[0]
    return False


def simple2complex(x, y):
    return x*50, y*50


def complex2simple(x, y):
    return int(float(x)/50), int(float(y)/50)




if __name__ == '__main__':
    Reversi = reversi()
    pygame.init()
    screen = pygame.display.set_mode((400, 400), 0, 32)
    pygame.display.set_caption('Reversi')
    background = pygame.image.load('./res/board.png')
    white = pygame.image.load('./res/white.png')
    black = pygame.image.load('./res/black.png')
    my_font = pygame.font.SysFont("arial", 30)
    for event in pygame.event.get():
        if event.type == QUIT:
            exit()
    screen.blit(background, (0, 0))
    screen.blit(white, (simple2complex(3, 3)))
    screen.blit(white, (simple2complex(4, 4)))
    screen.blit(black, (simple2complex(3, 4)))
    screen.blit(black, (simple2complex(4, 3)))
    pygame.display.update()
    flag = Reversi.feasible_move_show()
    old_flag = flag
    pos_weight=np.array([[7,2,5,4,4,5,2,7],[2,1,3,3,3,3,1,2],[5,3,6,5,5,5,3,5],[4,3,5,6,6,5,3,4],[4,3,5,6,6,5,3,4],[5,3,6,5,5,5,3,5],[2,1,3,3,3,3,1,2],[7,2,5,4,4,5,2,7]])
    model = raw_input('Battle with ai ? (y/n)')
    if model == 'y':
        first = raw_input('please choose who first (black first), if ai (y/n)?')
        if first == 'y':
            ai = -1
        else:
            ai = 1
    while old_flag or flag:
        old_flag = copy.copy(flag)
        flag = Reversi.feasible_move_show()
        if flag:
            if model == 'y' and Reversi.turn == ai:
                if quick_policy(flag):
                    ai_pos = quick_policy(flag)
                else:
                    state = Reversi.state.copy()
                    ai_pos = UCT_MCTS(state, 100, 5, ai,pos_weight)
                Reversi.move(ai_pos[0], ai_pos[1])

            else:
                flag_stop = False
                while not flag_stop:
                    pygame.event.set_allowed([MOUSEBUTTONDOWN])
                    event = pygame.event.wait()
                    if event.type == QUIT:
                        exit()
                    if event.type == MOUSEBUTTONDOWN:
                        x, y = event.pos
                        pos = complex2simple(x, y)
                        if Reversi.move(pos[1], pos[0]):
                            flag_stop = True
        else:
            if Reversi.turn == 1:
                text_surface = my_font.render(
                    "No place for white", True, (0, 0, 255))
                screen.blit(text_surface, (120, 150))
                pygame.display.update()
                time.sleep(0.5)
            elif Reversi.turn == -1:
                text_surface = my_font.render(
                    "No place for black", True, (0, 0, 255))
                screen.blit(text_surface, (120, 150))
                pygame.display.update()
                time.sleep(0.5)
        Reversi.plot_graph()
        Reversi.turn = -1*Reversi.turn
    Reversi.winorlose()
