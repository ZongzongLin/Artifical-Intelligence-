import pandas as pd
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
                math.sqrt(1.96*math.log(self.visits)/child.visits)
            if uctValue > bestValue:
                selected = child
                bestValue = uctValue
        return selected

    def update(self, result):
        self.visits += 1
        self.wins += 2*int(int(result) == int(self.activeplayer))-1.0

    def mostVisitedchild(self):
        mostVisited = self.children[0]
        for i in range(len(self.children)):
            if self.children[i].visits > mostVisited.visits:
                mostVisited = self.children[i]
        return mostVisited


def UCT_MCTS(state, max_Iteration, max_time, turn):
    blocksize = 50
    node_visited = 0.0
    root = TreeNode(None, state, None, turn)
    current_time = time.time()
    for i in range(blocksize):
        if time.time() > max_time + current_time:
            break
        node = root
        temp_state = copy.deepcopy(state)
        # selection
        while len(node.unreach) > 0 and len(node.children) > 0:
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
        actions = feasible_actions(temp_state, node.activeplayer)
        now_turn = copy.deepcopy(node.activeplayer)
        while len(actions) > 0:
            action = random.choice(actions)
            temp_state = update_state(
                temp_state, action[0], action[1], now_turn)
            node_visited += 1.0
            now_turn = -1*now_turn
            actions = feasible_actions(temp_state, now_turn)
        # backpropagation
        result = game_winner(temp_state)
        while(node):
            node.update(result)
            node = node.parent
    return root.mostVisitedchild().action


def game_winner(state):
    if sum((state == 1).apply(sum)) > sum((state == -1).apply(sum)):
        return 1
    elif sum((state == 1).apply(sum)) < sum((state == -1).apply(sum)):
        return -1
    else:
        return 0.5


def feasible_actions(state, turn):
    feasible_actions = []
    for x in range(8):
        for y in range(8):
            temp_state = copy.deepcopy(state)
            if state.iloc[x][y] == 0 and quick_check_state(temp_state, x, y, turn) == True:
                feasible_actions.append((x, y))
    return feasible_actions


def quick_check_state(state, x, y, turn):
    changer = []
    for i in range(y + 1, 8, 1):
        if state.iloc[x][i] == -1 * turn:
            changer.append(i)
            if i == 7:
                changer = []
                break
        elif state.iloc[x][i] == turn:
            break
        else:
            changer = []
            break
    if changer != []:
        return True
        # left
    changel = []
    for i in range(y - 1, -1, -1):
        if state.iloc[x][i] == -1 * turn:
            changel.append(i)
            if i == 0:
                changel = []
                break
        elif state.iloc[x][i] == turn:
            break
        else:
            changel = []
            break
    if changel != []:
        return True

        # down
    changed = []
    for i in range(x + 1, 8, 1):
        if state.iloc[i][y] == -1 * turn:
            changed.append(i)
            if i == 7:
                changed = []
                break
        elif state.iloc[i][y] == turn:
            break
        else:
            changed = []
            break
    if changed != []:
        return True

        # up
    changeu = []
    for i in range(x - 1, -1, -1):
        if state.iloc[i][y] == -1 * turn:
            changeu.append(i)
            if i == 0:
                changeu = []
                break
        elif state.iloc[i][y] == turn:
            break
        else:
            changeu = []
            break
    if changeu != []:
        return True

        # lu
    changelu = []
    for i in zip(range(x - 1, -1, -1), range(y - 1, -1, -1)):
        if state.iloc[i[0]][i[1]] == -1 * turn:
            changelu.append(i)
            if i[0] == 0 or i[1] == 0:
                changelu = []
                break
        elif state.iloc[i[0]][i[1]] == turn:
            break
        else:
            changelu = []
            break
    if changelu != []:
        return True
        # ld
    changeld = []
    for i in zip(range(x + 1, 8, 1), range(y - 1, -1, -1)):
        if state.iloc[i[0]][i[1]] == -1 * turn:
            changeld.append(i)
            if i[0] == 7 or i[1] == 0:
                changeld = []
                break
        elif state.iloc[i[0]][i[1]] == turn:
            break
        else:
            changeld = []
            break
    if changeld != []:
        return True
        # ru
    changeru = []
    for i in zip(range(x - 1, -1, -1), range(y + 1, 8, 1)):
        if state.iloc[i[0]][i[1]] == -1 * turn:
            changeru.append(i)
            if i[0] == 0 or i[1] == 7:
                changeru = []
                break
        elif state.iloc[i[0]][i[1]] == turn:
            break
        else:
            changeru = []
            break
    if changeru != []:
        return True
        # rd
    changerd = []
    for i in zip(range(x + 1, 8, 1), range(y + 1, 8, 1)):
        if state.iloc[i[0]][i[1]] == -1 * turn:
            changerd.append(i)
            if i[0] == 7 or i[1] == 7:
                changerd = []
                break
        elif state.iloc[i[0]][i[1]] == turn:
            break
        else:
            changerd = []
            break
    if changerd != []:
        return True
    return False


def update_state(state, x, y, turn):
    state.iloc[x][y] = turn
    # right
    changer = []
    for i in range(y + 1, 8, 1):
        if state.iloc[x][i] == -1 * turn:
            changer.append(i)
            if i == 7:
                changer = []
                break
        elif state.iloc[x][i] == turn:
            break
        else:
            changer = []
            break
    for i in changer:
        state.iloc[x][i] = turn
        # left
    changel = []
    for i in range(y - 1, -1, -1):
        if state.iloc[x][i] == -1 * turn:
            changel.append(i)
            if i == 0:
                changel = []
                break
        elif state.iloc[x][i] == turn:
            break
        else:
            changel = []
            break
    for i in changel:
        state.iloc[x][i] = turn

        # down
    changed = []
    for i in range(x + 1, 8, 1):
        if state.iloc[i][y] == -1 * turn:
            changed.append(i)
            if i == 7:
                changed = []
                break
        elif state.iloc[i][y] == turn:
            break
        else:
            changed = []
            break
    for i in changed:
        state.iloc[i][y] = turn

        # up
    changeu = []
    for i in range(x - 1, -1, -1):
        if state.iloc[i][y] == -1 * turn:
            changeu.append(i)
            if i == 0:
                changeu = []
                break
        elif state.iloc[i][y] == turn:
            break
        else:
            changeu = []
            break
    for i in changeu:
        state.iloc[i][y] = turn

        # lu
    changelu = []
    for i in zip(range(x - 1, -1, -1), range(y - 1, -1, -1)):
        if state.iloc[i[0]][i[1]] == -1 * turn:
            changelu.append(i)
            if i[0] == 0 or i[1] == 0:
                changelu = []
                break
        elif state.iloc[i[0]][i[1]] == turn:
            break
        else:
            changelu = []
            break
    for i in changelu:
        state.iloc[i[0]][i[1]] = turn
    # ld
    changeld = []
    for i in zip(range(x + 1, 8, 1), range(y - 1, -1, -1)):
        if state.iloc[i[0]][i[1]] == -1 * turn:
            changeld.append(i)
            if i[0] == 7 or i[1] == 0:
                changeld = []
                break
        elif state.iloc[i[0]][i[1]] == turn:
            break
        else:
            changeld = []
            break
    for i in changeld:
        state.iloc[i[0]][i[1]] = turn
    # ru
    changeru = []
    for i in zip(range(x - 1, -1, -1), range(y + 1, 8, 1)):
        if state.iloc[i[0]][i[1]] == -1 * turn:
            changeru.append(i)
            if i[0] == 0 or i[1] == 7:
                changeru = []
                break
        elif state.iloc[i[0]][i[1]] == turn:
            break
        else:
            changeru = []
            break
    for i in changeru:
        state.iloc[i[0]][i[1]] = turn
    # rd
    changerd = []
    for i in zip(range(x + 1, 8, 1), range(y + 1, 8, 1)):
        if state.iloc[i[0]][i[1]] == -1 * turn:
            changerd.append(i)
            if i[0] == 7 or i[1] == 7:
                changerd = []
                break
        elif state.iloc[i[0]][i[1]] == turn:
            break
        else:
            changerd = []
            break
    for i in changerd:
        state.iloc[i[0]][i[1]] = turn
    return state


class AI_MCTS:

    def __init__(self, state, turn):
        self.turn = turn
        self.state = state
        self.plays = {}
        self.wins = {}
        self.untouch = self.feasible_actions(self.state, self.turn)
        self.step_num = 64
        self.constant = 1.96
        self.total = 0.0
        self.max_time = 10
        self.feasible = copy.deepcopy(
            self.feasible_actions(self.state, self.turn))

    def get_action(self):
        if len(self.feasible) == 1:
            return self.feasible[0]
        start_time = time.time()
        simulation = 0
        while time.time()-start_time < self.max_time:
            first_turn = copy.deepcopy(self.turn)
            first_state = copy.deepcopy(self.state)
            self.process(first_state, first_turn)
            simulation += 1
        print simulation
        return self.Decision_move()

    def process(self, state, turn):
        node = TreeNode(self.state, self.feasible, self.turn)
        visited = set()
        player = turn
        move, Extension = self.selection(state, player)
        self.expansion((state, player, move), Extension)
        visited, winner = self.simulation(state, move, player, visited)
        self.backpropogation(visited, winner)

    def selection(self, state, play_turn):
        actions = self.feasible
        Flag_all = True
        if actions:
            for action in actions:
                if self.plays.get((state, play_turn, action,)) != False:
                    Flag_all = False
                    return action, True
            if Flag_all == True:
                max_value = -100000
                move = actions[0]
                for action in actions:
                    if self.wins[(play_turn, action)]/self.plays[(play_turn, action)]+math.sqrt(self.constant*math.log(self.total)/self.plays[(play_turn, action)]) > max_value:
                        move = action
                return move, False
        else:
            return (-1, -1)

    def expansion(self, tuple, Extension):
        if Extension and tuple not in self.plays:
            self.plays[tuple] = 0.0
            self.wins[tuple] = 0.0

    def simulation(self, state, move, player, visited):
        state = self.update_state(state, move[0], move[1], player)
        for i in range(self.step_num):
            player = -1*player
            wait_chose = self.feasible_actions(state, player)
            if wait_chose:
                if len(wait_chose) == 1:
                    move = wait_chose[0]
                else:
                    if [0, 0] in wait_chose:
                        move = [0, 0]
                    elif [0, 7] in wait_chose:
                        move = [0, 7]
                    elif [7, 0] in wait_chose:
                        move = [7, 0]
                    elif [7, 7] in wait_chose:
                        move = [7, 7]
                    else:
                        move = random.choice(wait_chose)
                state = self.update_state(state, move[0], move[1], player)
                visited = +1
            else:
                move = (-1, -1)
            if self.game_is_end(state, player):
                break
        winner = self.game_winner(state)
        return visited, winner

    def backpropogation(self, visited, winner):
        if (player, move) in self.plays:
            self.plays[(player, move)] += 1.0
            if winner == 0.5:
                self.wins[(player, move)] += 0.5
            elif player == winner:
                self.wins[(player, move)] += 1.0
        self.total += 1.0

    def game_is_end(self, state, turn):
        if not self.feasible_actions(state, turn) and not self.feasible_actions(state, -1 * turn):
            return True
        else:
            return False

    def game_winner(self, state):
        if sum((state == 1).apply(sum)) > sum((state == -1).apply(sum)):
            return 1
        elif sum((state == 1).apply(sum)) < sum((state == -1).apply(sum)):
            return -1
        else:
            return 0.5

    def Decision_move(self):
        wait = [(self.plays.get((self.turn, action)), action)
                for action in self.feasible]
        wait.sort(key=lambda x: x[0], reverse=True)
        return wait[0][1]


class reversi(object):

    def __init__(self):
        self.turn = -1
        self.victory = 0
        self.state = pd.DataFrame(0, index=map(lambda x: str(x),
                                               range(8)), columns=map(lambda x: str(x),
                                                                      range(8)))
        self.state.iloc[3][3] = 1
        self.state.iloc[3][4] = -1
        self.state.iloc[4][3] = -1
        self.state.iloc[4][4] = 1

    def move(self, x, y):
        return self.update_state(x, y)

    def feasible_move_show(self):
        feasible_move = []
        for x in range(8):
            for y in range(8):
                temp_state = copy.deepcopy(self.state)
                if self.state.iloc[x][y] == 0 and type(self.complete_update_state(temp_state, x, y, self.turn)) == type(pd.DataFrame()):
                    feasible_move.append((x, y))
        return feasible_move

    def update_state(self, x, y):
        if (x, y) not in self.feasible_move_show():
            return False
        self.state.iloc[x][y] = self.turn  # upated the change
        self.state = self.complete_update_state(self.state, x, y, self.turn)
        return True

    def complete_update_state(self, state, x, y, turn):
        # right
        changer = []
        for i in range(y+1, 8, 1):
            if state.iloc[x][i] == -1*turn:
                changer.append(i)
                if i == 7:
                    changer = []
                    break
            elif state.iloc[x][i] == turn:
                break
            else:
                changer = []
                break
        for i in changer:
            state.iloc[x][i] = turn
        # left
        changel = []
        for i in range(y-1, -1, -1):
            if state.iloc[x][i] == -1*turn:
                changel.append(i)
                if i == 0:
                    changel = []
                    break
            elif state.iloc[x][i] == turn:
                break
            else:
                changel = []
                break
        for i in changel:
            state.iloc[x][i] = turn

        # down
        changed = []
        for i in range(x+1, 8, 1):
            if state.iloc[i][y] == -1*turn:
                changed.append(i)
                if i == 7:
                    changed = []
                    break
            elif state.iloc[i][y] == turn:
                break
            else:
                changed = []
                break
        for i in changed:
            state.iloc[i][y] = turn

        # up
        changeu = []
        for i in range(x-1, -1, -1):
            if state.iloc[i][y] == -1*turn:
                changeu.append(i)
                if i == 0:
                    changeu = []
                    break
            elif state.iloc[i][y] == turn:
                break
            else:
                changeu = []
                break
        for i in changeu:
            state.iloc[i][y] = turn

        # lu
        changelu = []
        for i in zip(range(x-1, -1, -1), range(y-1, -1, -1)):
            if state.iloc[i[0]][i[1]] == -1*turn:
                changelu.append(i)
                if i[0] == 0 or i[1] == 0:
                    changelu = []
                    break
            elif state.iloc[i[0]][i[1]] == turn:
                break
            else:
                changelu = []
                break
        for i in changelu:
            state.iloc[i[0]][i[1]] = turn
        # ld
        changeld = []
        for i in zip(range(x+1, 8, 1), range(y-1, -1, -1)):
            if state.iloc[i[0]][i[1]] == -1*turn:
                changeld.append(i)
                if i[0] == 7 or i[1] == 0:
                    changeld = []
                    break
            elif state.iloc[i[0]][i[1]] == turn:
                break
            else:
                changeld = []
                break
        for i in changeld:
            state.iloc[i[0]][i[1]] = turn
        # ru
        changeru = []
        for i in zip(range(x-1, -1, -1), range(y+1, 8, 1)):
            if state.iloc[i[0]][i[1]] == -1*turn:
                changeru.append(i)
                if i[0] == 0 or i[1] == 7:
                    changeru = []
                    break
            elif state.iloc[i[0]][i[1]] == turn:
                break
            else:
                changeru = []
                break
        for i in changeru:
            state.iloc[i[0]][i[1]] = turn
        # rd
        changerd = []
        for i in zip(range(x+1, 8, 1), range(y+1, 8, 1)):
            if state.iloc[i[0]][i[1]] == -1*turn:
                changerd.append(i)
                if i[0] == 7 or i[1] == 7:
                    changerd = []
                    break
            elif state.iloc[i[0]][i[1]] == turn:
                break
            else:
                changerd = []
                break
        for i in changerd:
            state.iloc[i[0]][i[1]] = turn

        #print [changel, changer, changed, changeu, changeld, changelu, changerd, changeru],(x,y)
        if len(changel+changer+changed+changeu+changeld+changelu+changerd+changeru) == 0:
            return False
        else:
            return state

    def winorlose(self):
        if sum((self.state == 1).apply(sum)) < sum((self.state == -1).apply(sum)):
            text_surface = my_font.render("Black win", True, (0, 0, 255))
            screen.blit(text_surface, (120, 150))
            pygame.display.update()
            time.sleep(10)
            print 'Black win'
        elif sum((self.state == 1).apply(sum)) > sum((self.state == -1).apply(sum)):
            text_surface = my_font.render("White win", True, (0, 0, 255))
            screen.blit(text_surface, (120, 150))
            pygame.display.update()
            time.sleep(10)
            print 'White win'
        elif sum((self.state == 1).apply(sum)) == sum((self.state == -1).apply(sum)):
            text_surface = my_font.render("Double win", True, (0, 0, 255))
            screen.blit(text_surface, (120, 150))
            pygame.display.update()
            time.sleep(10)
            print 'Double win'


def quick_policy(flag):
    if len(flag) == 1:
        return flag[0]
    return False


def simple2complex(x, y):
    return x*50, y*50


def complex2simple(x, y):
    return int(float(x)/50), int(float(y)/50)


def plot_graph(state):
    screen.blit(backgroud, (0, 0))
    for i in range(8):
        for j in range(8):
            if state.iloc[i][j] == 1:
                screen.blit(white, (simple2complex(j, i)))
            elif state.iloc[i][j] == -1:
                screen.blit(black, (simple2complex(j, i)))
    pygame.display.update()

if __name__ == '__main__':
    Reversi = reversi()
    pygame.init()
    screen = pygame.display.set_mode((400, 400), 0, 32)
    pygame.display.set_caption('Reversi')
    backgroud = pygame.image.load('./res/board.png')
    white = pygame.image.load('./res/white.png')
    black = pygame.image.load('./res/black.png')
    my_font = pygame.font.SysFont("arial", 30)
    for event in pygame.event.get():
        if event.type == QUIT:
            exit()
    screen.blit(backgroud, (0, 0))
    screen.blit(white, (simple2complex(3, 3)))
    screen.blit(white, (simple2complex(4, 4)))
    screen.blit(black, (simple2complex(3, 4)))
    screen.blit(black, (simple2complex(4, 3)))
    pygame.display.update()
    flag = Reversi.feasible_move_show()
    old_flag = flag
    model = raw_input('Battle with ai ? (y/n)')
    if model == 'y':
        first = raw_input('plase choose who first (white first), if ai (y/n)?')
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
                    state = copy.deepcopy(Reversi.state)
                    ai_pos = UCT_MCTS(state, 100, 10, ai)
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
                time.sleep(1)
            elif Reversi.turn == -1:
                text_surface = my_font.render(
                    "No place for black", True, (0, 0, 255))
                screen.blit(text_surface, (120, 150))
                pygame.display.update()
                time.sleep(1)
        plot_graph(Reversi.state)
        Reversi.turn = -1*Reversi.turn
    Reversi.winorlose()
