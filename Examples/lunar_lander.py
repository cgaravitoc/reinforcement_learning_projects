from Utils.train import TrainRun
from Agents.agentsCS import SarsaCS, nStepCS
from Agents.linearQ import TilesQ
from Utils.interpreters import gym_interpreter1
import matplotlib.pyplot as plt
import pandas as pd
from Utils.utils import Plot



def try_env():
    '''
    Loads an agent and runs it
    without learning on the
    Lunar lander environment
    '''
    print('Loading agent and environment...')
     # Create agent
    agent = load_agent_SarsaCS()
    # Create train-and-run object
    act = load_act(agent, 'Sarsa')
    # Show the untrained agent
    print('Showing the untrained agent...')
    act.run(visual=True)
    print('Done!')


def train_and_run_SARSA():
    '''
    Trains a SARSA agent on the Mountain Car
    '''
    # Create agent
    print('Loading agent and environment...')
    agent = load_agent_SarsaCS()
    # Create train-and-run object
    act = load_act(agent, 'Sarsa')
    # Train the agent
    print('Training the agent...')
    act.train()
    # Show the trained agent
    print('Showing the trained agent...')
    act.run()
    # Testing the agent
    print('Testing the agent...')
    act.test()
    print('Done!')

def sweep_SARSA():
    '''
    Runs a sweep over alpha
    '''
    # Create agent
    print('Loading agent and environment...')
    agent = load_agent_SarsaCS()
    # Create train-and-run object
    act = load_act(agent, 'Sarsa')
    # Sweep alpha
    print('Sweeping alpha...')
    alphas = [0.2/8, 0.4/8, 0.8/8]
    act.sweep(parameter='alpha', values=alphas, num_simulations=10)
    print('Done!')


def load_agent_SarsaCS() -> SarsaCS:
    '''
    Creates a SarsaCS agent with a set
    of parameters determined inset
    '''
    # Define parameters
    parameters = {"numDims":8,\
                  "nA":4,\
                  "gamma":1,\
                  "epsilon":0.1,\
                  "alpha":0.1,\
                  "numTilings":8,\
                  "numTiles":[10, 10,10, 10,10, 10,10, 10],\
                  "scaleFactors":[\
                    {"min":-90., "max":90.}, # x coordiantes
                    {"min":-90., "max":90.}, # y coordiantes
                    {"min":-5., "max":5.}, # x velocity
                    {"min":-5., "max":5.}, # y velocity
                    {"min":-3.1415927, "max":3.1415927}, # object angle
                    {"min":-5., "max":5.}, # angular velocity
                    {"min":0., "max":1.}, # boolean leg 1
                    {"min":-0., "max":1.}, # boolean leg 2
                    ]
                    }
    # Create approximating function
    Q = TilesQ(parameters=parameters)
    # Create agent
    return SarsaCS(parameters, Q)


def load_agent_nStepCS() -> nStepCS:
    '''
    Creates a nStepCS agent with a set
    of parameters determined inset
    '''
    # Define parameters
    parameters = {"numDims":8,\
                  "nA":4,\
                  "gamma":1,\
                  "epsilon":0.1,\
                  "alpha":0.1,\
                  "numTilings":8,\
                  "numTiles":[10, 10,10, 10,10, 10,10, 10],\
                  "n":2,\
                  "scaleFactors":[\
                    {"min":-90., "max":90.}, # x coordiantes
                    {"min":-90., "max":90.}, # y coordiantes
                    {"min":-5., "max":5.}, # x velocity
                    {"min":-5., "max":5.}, # y velocity
                    {"min":-3.1415927, "max":3.1415927}, # object angle
                    {"min":-5., "max":5.}, # angular velocity
                    {"min":0., "max":1.}, # boolean leg 1
                    {"min":-0., "max":1.}, # boolean leg 2
                    ]
                    }
    # Create approximating function
    Q = TilesQ(parameters=parameters)
    # Create agent
    return nStepCS(parameters, Q)



def load_act(agent, model_name:str) -> TrainRun:
    '''
    Creates a train-and-run object with 
    parameters given inset
    '''
    act = TrainRun(\
        env_name = 'LunarLander-v2',\
        state_interpreter=gym_interpreter1,\
        agent=agent,\
        model_name=model_name,\
        num_rounds=1000, # by default was in 1000
        num_episodes=200
        )
    return act

def sweep_nStep():
    '''
    Runs a sweep over alpha
    '''
    # Create agent
    print('Loading agent and environment...')
    agent = load_agent_nStepCS()
    # Create train-and-run object
    act = load_act(agent, 'nStep')
    # Sweep alpha
    print('Sweeping alpha...')
    alphas = [0.2/8, 0.4/8, 0.8/8]
    n = [2, 4, 8]
    act.sweep2(parameter1='alpha', values1=alphas, parameter2='n', values2=n, num_simulations=10)
    print('Done!')


def train_and_compare():
    # Create agent
    agent_SARSA = load_agent_SarsaCS()
    # Create train-and-run object
    act = load_act(agent_SARSA, 'Sarsa')
    # Train the SARSA agent
    print('Training SARSA agent...')
    act.train()
    # Testing the agent
    print('Testing SARSA agent...')
    act.num_episodes = 100
    act.test(to_df=True)
    df_sarsa = act.data
    # Create agent
    agent_nSARSA = load_agent_nStepCS()
    # Create train-and-run object
    act = load_act(agent_nSARSA, 'nSarsa')
    # Train the agent
    print('Training nSARSA agent...')
    act.train()
    # Testing the agent
    print('Testing nSARSA agent...')
    act.num_episodes = 100
    act.test(to_df=True)
    df_nsarsa = act.data
    # Compare performances
    df = pd.concat([df_sarsa, df_nsarsa], ignore_index=True)
    p = Plot(df)
    p.plot_histogram_rewards(act.file_compare_hist)
    print(f'Plot saved to {act.file_compare_hist}')
    p.plot_rewards(act.file_compare_rew)
    print(f'Plot saved to {act.file_compare_rew}')