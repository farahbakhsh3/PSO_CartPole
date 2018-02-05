import gym
import numpy as np
from random import random


class particle:
    pass


env = gym.make('CartPole-v0')
dimensions = 4
SwarmSize = 5
_w = 1
wDamp = .9
c1 = 2
c2 = 2
maxIter = 100

globalbest = particle()
globalbest.bestfitness = -1
globalbest.bestposition = []
particles = []

def Fit(p):
    # print(p)
    length = []
    for j in range(100):
        observation = env.reset()
        done = False
        cnt = 0
        while not done:
            cnt += 1
            action = 1 if np.dot(observation, p) > 0 else 0
            # env.render()
            observation, reward, done, info = env.step(action)
            if done:
                break
        length.append(cnt)
    average_length = np.average(length)
    return average_length

def PSO():
    w = _w
    p = particle()
    for _ in range(SwarmSize):
        p.position = np.random.uniform(0, 1, dimensions)
        p.velocity = np.random.uniform(0, 1, dimensions)
        p.fitness = Fit(p.position)
        p.bestposition = p.position
        p.bestfitness = p.fitness
        if p.bestfitness > globalbest.bestfitness:
            globalbest.bestfitness = p.bestfitness
            globalbest.bestposition = p.bestposition.copy()
        particles.append(p)

    for i in range(maxIter):
        for pp in particles:
            pp.velocity = w * pp.velocity + \
                          c1 * random() * (pp.bestposition - pp.position) + \
                          c2 * random() * (globalbest.bestposition - pp.position)
            x = pp.position + pp.velocity
            for _ in range(dimensions):
                if x[_] > 1: x[_] = random()
                if x[_] < 0: x[_] = random()
            pp.position = x
            pp.fitness = Fit(pp.position)
            if pp.fitness > pp.bestfitness:
                pp.bestposition = pp.position
                pp.bestfitness = pp.fitness
                if pp.bestfitness > globalbest.bestfitness:
                    globalbest.bestfitness = pp.bestfitness
                    globalbest.bestposition = pp.bestposition
        w = w * wDamp
        print(i, globalbest.bestfitness, globalbest.bestposition)
        if globalbest.bestfitness == 200:
            break

PSO()
print(globalbest.bestposition)
print (globalbest.bestfitness)

best_weights = globalbest.bestposition
ave_len = []
# env = wrappers.Monitor(env, 'new', force=True)
for _ in range(10):
    cnt = 0
    done = False
    observation = env.reset()
    while not done:
        env.render()
        cnt += 1
        action = 1 if np.dot(observation, best_weights) > 0 else 0
        observation, reward, done, info = env.step(action)
        if done:
            break
    print('with best weights game ', _, ' lasted: ', cnt, 'moves')
    ave_len.append(cnt)
print(' ')
print(' -------------- ')
print(' ')
print('with best weights game average lasted: ', np.average(ave_len), 'moves')
env.close()
