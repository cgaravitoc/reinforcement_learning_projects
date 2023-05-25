from Utils.train import TrainRun
from Agents.agentsCS import SarsaCS, nStepCS
from Agents.linearQ import TilesQ
from Utils.interpreters import gym_interpreter1


def try_env():
    '''
    Loads an agent and runs it
    without learning on the
    Mountain Car environment
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
    act.sweep(parameter='alpha', values=alphas)
    print('Done!')


def load_agent_SarsaCS() -> SarsaCS:
    '''
    Creates a SarsaCS agent with a set
    of parameters determined inset
    '''
    # Define parameters
    parameters = {"numDims":2,\
                  "nA":3,\
                  "gamma":1,\
                  "epsilon":0.1,\
                  "alpha":0.1,\
                  "numTilings":8,\
                  "numTiles":[10, 10],\
                  "scaleFactors":[\
                    {"min":-1.2,\
                    "max":0.6},
                    {"min":-0.07,\
                      "max":0.07}]
                    }
    # Create approximating function
    Q = TilesQ(parameters=parameters)
    # Create agent
    return SarsaCS(parameters, Q)


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
        num_rounds=1000,\
        num_episodes=200
        )
    return act


def sweep_nStep():
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
    n = [2, 4, 8]
    act.sweep2(parameter1='alpha', values1=alphas, parameter2='n', values2=n)
    print('Done!')