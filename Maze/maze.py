from PIL import Image
import numpy as np
import pandas as pd
import random
import pickle as pickle
from PIL import Image,ImageDraw
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import copy
import random
import math

gamma=0.99
def takeaction(state,action):
    if action=='L':
        nextstate=(state[0],state[1]-1)
    elif action=='U':
        nextstate = (state[0]-1, state[1])
    elif action=='R':
        nextstate = (state[0] , state[1]+1)
    elif action=='D':
        nextstate = (state[0]+1, state[1])
    return nextstate

def takeactionsimple(state,action,state_action_dict):
    if action=='L':
        nextstate=(state[0],state[1]-1)
        while nextstate not in state_action_dict.keys():
            nextstate = (nextstate[0], nextstate[1] - 1)
    elif action=='U':
        nextstate = (state[0]-1, state[1])
        while nextstate not in state_action_dict.keys():
            nextstate = (nextstate[0]-1, nextstate[1] )
    elif action=='R':
        nextstate = (state[0] , state[1]+1)
        while nextstate not in state_action_dict.keys():
            nextstate = (nextstate[0], nextstate[1] + 1)
    elif action=='D':
        nextstate = (state[0]+1, state[1])
        while nextstate not in state_action_dict.keys():
            nextstate = (nextstate[0]+1, nextstate[1])
    return nextstate



def takeaction_criteria(state,action,newpix):
    nextstate=takeaction(state,action)
    if nextstate[0]>48 or nextstate[1]>64 or nextstate[0]<0 or nextstate[1]<0 or newpix[nextstate[0]][nextstate[1]]==0:
        return False
    else:
        return True

def initial(destination):
    im = Image.open('maze.jpg')
    im=im.convert('L')
    im = im.crop((50, 48, 570, 440))
    width, high = im.size  # w:520,#h:392
    pix = np.array(im)
    newpix = np.zeros([49, 65])
    square = 8
    width_batch = int(width/square)
    high_batch = int(high/square)
    feasiblestates=[]
    for i in range(high_batch):
        for j in range(width_batch):
            if pix[i*square+4][j*square+4]<200:
                newpix[i][j] = 1
                feasiblestates.append((i,j))
            else:
                newpix[i][j] = 0
    newpix[destination[0]][destination[1]]=1
    feasiblestates.append(destination)
    actions=['R','D','L','U']
    end_nodes=[]
    oppodirection = [['L', 'R'], ['U', 'D'], ['R', 'L'], ['D', 'U']]
    state_action_normal_dict={}
    state_action_dict={}
    for state in feasiblestates:
        action_take=[]
        for action in actions:
            if  takeaction_criteria(state,action,newpix)==True:
                action_take.append(action)
        if len(action_take)==1:
            end_nodes.append(state)
        if len(action_take)==2 and action_take in oppodirection and state!=destination:
            state_action_normal_dict[state]=action_take
        else:
            state_action_dict[state]=action_take
    simplestates=state_action_dict.keys()
    simplestates_str=map(lambda x:str(x),simplestates)
    Qmatrix=pd.DataFrame(0,index=simplestates_str,columns=actions)
    Rmatrix=pd.DataFrame(-1,index=simplestates_str,columns=['N'])
    for state in simplestates:
        if state!=destination:
            if state in end_nodes:
                Rmatrix.loc[str(state)]['N']=-1000
        else:
            Rmatrix.loc[str(state)]['N'] =1000
    return Rmatrix,state_action_dict,state_action_normal_dict,simplestates,simplestates_str,feasiblestates,Qmatrix,end_nodes

def random2choice(state,actions,Qmatrix):
    wait_choose=[]
    for action in actions:
        wait_choose.append(Qmatrix.loc[str(state)][action])
    wait_choose=map(lambda x:math.exp(0.1*x),wait_choose)
    sums=sum(wait_choose)
    wait_normalize=map(lambda x:x*1.0/sums,wait_choose)
    prob_actions = zip(wait_normalize, actions)
    prob_actions = filter(lambda x: x[1] != 0, prob_actions)
    cumulative_possible = 0.0
    q = random.uniform(0, 1)
    for prob_action in prob_actions:
        cumulative_possible = cumulative_possible + prob_action[0]
        if q < cumulative_possible:
            return prob_action[1]


def train(Rmatrix,state_action_dict,simplestates,Qmatrix,end_nodes,destination):
    simplestates.sort(key=lambda x: x[1]+x[0], reverse=True)
    for i in range(5000):
        if i<500:
            start=simplestates[i%len(simplestates)]
        else:
            start=random.choice(simplestates)
        current_state=start
        step=0
        while current_state!=destination:
            if current_state in end_nodes:
                action=state_action_dict[current_state][0]
            else:
                action=random2choice(current_state,state_action_dict[current_state],Qmatrix)
            next_state=takeactionsimple(current_state,action,state_action_dict)
            future_reward=[]
            for next_action in state_action_dict[next_state]:
                future_reward.append(Qmatrix.loc[str(next_state)][next_action])
            Qmatrix.loc[str(current_state)][action]=Rmatrix.loc[str(next_state)]['N']+gamma*max(future_reward)
            current_state=next_state
            step=step+1
        print'complete train %d /10000 times'%(i+1)
    output = open('Qmatrix.pkl', 'wb')
    pickle.dump(Qmatrix, output)
    output.close()
    return Qmatrix

def nearest_best(state,Rmatrix,state_action_normal_dict,state_action_dict):
    if state in state_action_normal_dict.keys():
        direction0 = takeactionsimple(state,state_action_normal_dict[state][0], state_action_dict)
        direction1 = takeactionsimple(state, state_action_normal_dict[state][1], state_action_dict)
        if Rmatrix.loc[str(direction0)]['N']>Rmatrix.loc[str(direction1)]['N']:
            return direction0
        else:
            return direction1
    return state


def simple2complex(x):
    return (x[0]*8+4,x[1]*8+4)

def complex2simple(x):
    return (int(x[0]/8),int(x[1]/8))

def plotreverse(x):
    return (x[1],x[0])

def modify(line):
    vecdot=(line[1][0]-line[0][0])*(line[2][0]-line[1][0])+(line[1][1]-line[0][1])*(line[2][1]-line[1][1])
    if vecdot<0:
        return [line[0]]+line[2:]
    else:
        return  reduce(lambda x, y: x if y in x else x + y,[],line)


def main():
    lena = mpimg.imread('deal_maze.jpg')
    plt.imshow(lena)
    plt.axis('off')
    plt.show()
    a=raw_input('Insert the begin for first pix:')
    b=raw_input('Insert the begin for second pix:')
    x=(int(b),int(a))
    current_state=complex2simple(x)
    destination = (44, 63)
    Rmatrix, state_action_dict, state_action_normal_dict, simplestates, simplestates_str, feasiblestates,Qmatrix,end_nodes=initial(destination)
    print('complete initial')
    flag=1
    while flag==1:
        if current_state not in feasiblestates:
            print 'Wrong initial'
            lena = mpimg.imread('deal_maze.jpg')
            plt.imshow(lena)
            plt.axis('off')
            plt.show()
            a = raw_input('Insert the begin for first pix:')
            b = raw_input('Insert the begin for second pix:')
            x = (int(b), int(a))
            current_state = complex2simple(x)
        else:
            flag=0
    door_for_template=raw_input('Use the Trained Q matrix ?[y/n]:')
    if door_for_template=='y':
        fr = open('Qmatrix.pkl')
        Qmatrix = pickle.load(fr)
        fr.close()
    else:
        Qmatrix=train(Rmatrix,state_action_dict,simplestates,Qmatrix,end_nodes,destination)
        print('Complete train')
    line=[current_state]
    current_state_simple=nearest_best(current_state,Rmatrix,state_action_normal_dict,state_action_dict)
    line.append(current_state_simple)
    current_state=current_state_simple
    for step in range(1000):
        action_wait=zip(Qmatrix.loc[str(current_state)],Qmatrix.columns)
        action_wait=filter(lambda x:x[1] in state_action_dict[current_state],action_wait)
        action_wait.sort(key=lambda x: x[0], reverse=True)
        next_state = takeactionsimple(current_state, action_wait[0][1], state_action_dict)
        current_state = next_state
        line.append(current_state)
        if current_state==destination:
            print('Find the destination!')
            break
    if len(line)>=1001:
        print line
        print('Could not reach the destination!')
    else:
        print line
        line=modify(line)
        print 'Length of line:',len(line)
    line=map(lambda x:simple2complex(x),line)
    line=map(lambda x:plotreverse(x),line)
    img = Image.open('deal_maze.jpg')
    img_d = ImageDraw.Draw(img)
    img_d.line(line,fill=(255,0,0),width=5)
    img.show()
    img.save('./mazeresult.png')

if __name__=='__main__':
    main()













