from Utils.train import TrainRun
from Utils.utils import Plot
from Agents.agents import SARSA
from Agents.agentsCS import OnlineQN, DQN
from Agents.deepQ import FFNQ, FFNQ_D
from Utils.interpreters import gridW_nS_interpreter, gridW_vector_interpreter
import matplotlib.pyplot as plt
import pandas as pd

def test():
    from Environments.GridWorld.GridWorld import Gridworld
    grid_size = 10
    mode = 'static'
    env = Gridworld(size=grid_size, mode=mode)
    env.render()
    plt.close()


def train_and_run():
    '''
    Trains a SARSA agent on the Grid World
    '''
    # Define parameters
    grid_size = 4
    parameters = {"nS":grid_size**8,\
                  "nA":4,\
                  "gamma":1,\
                  "epsilon":0.1,\
                  "alpha":0.1,\
                    }
    # Create agent
    agent = SARSA(parameters)
    # Create train-and-run object
    act = TrainRun(\
        env_name = f'Gridworld-{grid_size}-static',\
        # env_name = f'Gridworld-{grid_size}-player',\
        # env_name = f'Gridworld-{grid_size}-random',\
        state_interpreter=gridW_nS_interpreter,\
        agent=agent,\
        model_name='Sarsa',\
        num_rounds=1000,\
        num_episodes=1000
        )
    # Train the agent
    print('Training the agent...')
    act.train()
    # # Show the trained agent
    # print('Showing the trained agent...')
    # act.run()
    # Testing the agent
    act.test()
    print('Done!')


def train_and_run_OnlineQN():
    '''
    Trains an Online DeepQN agent on the Grid World
    '''
    # Define parameters
    grid_size = 4
    parameters = {"numDims":(grid_size**2)*4,\
                  "nA":4,\
                  "gamma":0.99,\
                  "epsilon":0.1,\
                  "alpha":1e-3,\
                  "input_size":(grid_size**2)*4,\
                  "hidden_size":64,\
                  "output_size":4
                    }
    # Create neural network
    Q = FFNQ(parameters=parameters)
    # Create agent
    agent = OnlineQN(parameters, Q)
    # Create train-and-run object
    act = TrainRun(\
        # env_name = f'Gridworld-{grid_size}-static',\
        env_name = f'Gridworld-{grid_size}-player',\
        # env_name = f'Gridworld-{grid_size}-random',\
        state_interpreter=gridW_vector_interpreter,\
        agent=agent,\
        model_name='OnlineQN',\
        num_rounds=1000,\
        num_episodes=1000
        )
    # Train the agent
    print('Training the agent...')
    act.train()
    # Show the trained agent
    # print('Showing the agent...')
    # act.agent.epsilon = 0
    # act.num_rounds = 15
    # act.run(visual=False)
    # Testing the agent
    print('Testing the agent...')
    act.num_episodes = 100
    act.test()
    print('Done!')


def train_and_compare():
    # Define parameters SARSA
    grid_size = 4
    parameters = {"nS":grid_size**8,\
                  "nA":4,\
                  "gamma":0.99,\
                  "epsilon":0.1,\
                  "alpha":0.1,\
                    }
    # Create agent
    agent_SARSA = SARSA(parameters)
    # Create train-and-run object
    act = TrainRun(\
        # env_name = f'Gridworld-{grid_size}-static',\
        env_name = f'Gridworld-{grid_size}-player',\
        # env_name = f'Gridworld-{grid_size}-random',\
        state_interpreter=gridW_nS_interpreter,\
        agent=agent_SARSA,\
        model_name='Sarsa',\
        num_rounds=1000,\
        num_episodes=1000
        )
    # Train the SARSA agent
    print('Training SARSA agent...')
    act.train()
    # Testing the agent
    print('Testing SARSA agent...')
    act.num_episodes = 100
    act.test(to_df=True)
    df_sarsa = act.data
    # Define parameters
    parameters = {"numDims":(grid_size**2)*4,\
                  "nA":4,\
                  "gamma":0.99,\
                  "epsilon":0.1,\
                  "alpha":1e-3,\
                  "input_size":(grid_size**2)*4,\
                  "hidden_size":64,\
                  "output_size":4
                    }
    # Create neural network
    Q = FFNQ(parameters=parameters)
    # Create agent
    agent_DQN = OnlineQN(parameters, Q)
    # Create train-and-run object
    act = TrainRun(\
        # env_name = f'Gridworld-{grid_size}-static',\
        env_name = f'Gridworld-{grid_size}-player',\
        # env_name = f'Gridworld-{grid_size}-random',\
        state_interpreter=gridW_vector_interpreter,\
        agent=agent_DQN,\
        model_name='OnlineQN',\
        num_rounds=1000,\
        num_episodes=1000
        )
    # Train the agent
    print('Training OnlineQN agent...')
    act.train()
    # Testing the agent
    print('Testing OnlineQN agent...')
    act.num_episodes = 100
    act.test(to_df=True)
    df_qn = act.data
    # Compare performances
    df = pd.concat([df_sarsa, df_qn], ignore_index=True)
    p = Plot(df)
    p.plot_histogram_rewards(act.file_compare_hist)
    print(f'Plot saved to {act.file_compare_hist}')
    p.plot_rewards(act.file_compare_rew)
    print(f'Plot saved to {act.file_compare_rew}')



def train_and_run_DQN():
    '''
    Trains a DeepQ Network on the Grid World
    '''
    # Define parameters
    grid_size = 4
    parameters = {"numDims":(grid_size**2)*4,\
                  "nA":4,\
                  "gamma":0.99,\
                  "epsilon":0.05,\
                  "alpha":1e-5,\
                  "input_size":(grid_size**2)*4,\
                  "hidden_size_1":254,\
                  "hidden_size_2":64,\
                  "hidden_size_3":64,\
                  "output_size":4,\
                  "c": 4,\
                  "len_sample": 2
                    }
    # Create neural network
    Q = FFNQ_D(parameters=parameters)
    # Create agent
    agent = DQN(parameters, Q)
    # Create train-and-run object
    act = TrainRun(\
        # env_name = f'Gridworld-{grid_size}-static',\
        # env_name = f'Gridworld-{grid_size}-player',\
        env_name = f'Gridworld-{grid_size}-random',\
        state_interpreter=gridW_vector_interpreter,\
        agent=agent,\
        model_name='DQN',\
        num_rounds=1000,\
        num_episodes=1000
        )
    # Train the agent
    print('Training the agent...')
    act.train()
    # Show the trained agent
    # print('Showing the agent...')
    # act.agent.epsilon = 0
    # act.num_rounds = 15
    # act.run(visual=False)
    # Testing the agent
    print('Testing the agent...')
    act.num_episodes = 100
    act.test()
    print('Done!')