import Utils.utils as utils
from os import path
import gymnasium as gym
import Environments as E
from Environments.GridWorld.GridWorld import Gridworld
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class TrainRun :
    '''
    Class to train and run an agent in an environment.
    '''
    def __init__(self,\
                env_name:str,\
                state_interpreter,\
                agent,\
                model_name:str,\
                num_rounds:int,\
                num_episodes:int 
                ) -> None:
        self.env_name = env_name
        self.state_interpreter = state_interpreter
        self.agent = agent
        self.num_rounds = num_rounds
        self.num_episodes = num_episodes
        self.model_name = model_name
        file_name = (f'{model_name}_in_{env_name}')
        self.file_model = path.join('models', f'{file_name}.pt')
        self.file_csv = path.join('data', f'{file_name}.csv')
        self.file_png = path.join('images', f'{file_name}.png')
        self.file_losses = path.join('images', f'{file_name}_losses.png')
        self.file_test = path.join('images', f'{file_name}_test.png')
        self.file_compare_hist = path.join('images', f'comparison_hist.png')
        self.file_compare_rew = path.join('images', f'comparison_rew.png')

    def load_env(self, render_mode):
        '''
        Loads environment. If using gymnasium environments, render mode
        is different for training (None) than for running (rgb_array). Render
        mode can only set when instantiating object.
        '''
        if self.env_name in ['MountainCar-v0']:
            self.environment = gym.make('MountainCar-v0', render_mode=render_mode)
        elif self.env_name in ['LunarLander-v2']:
            self.environment = gym.make('LunarLander-v2', render_mode=render_mode)
        elif 'Gridworld' in self.env_name:
           size = int(self.env_name.split('-')[1])
           mode = self.env_name.split('-')[2]
           self.environment = Gridworld(size=size, mode=mode)
        else:
            raise Exception('Unknown environment. Please modify TrainRun.load_env() to include it.')

    def save_agent(self):
        '''
        Saves agent model to a file
        '''
        pass
    
    def run(self, visual=True, learn=False, num_rounds=200):
        '''
        Runs the agent on the environment and displays the behavior.
        Input:
          - visual,
            True: displays the environment as in a video using environment render
            False: displays the behavioral data in the terminal step by step
        '''
        if visual:
          # Displays the environment as in a video
          # Creates environment
          self.load_env(render_mode='human')
          # Creates episode
          episode = utils.Episode(environment=self.environment,\
                  agent=self.agent,\
                  model_name=self.model_name,\
                  num_rounds=num_rounds,\
                  state_interpreter=self.state_interpreter
                  )
          episode.renderize(learn=learn)
        else:
          # Displays data information in the terminal
          self.load_env(render_mode=None)
          # Creates episode
          episode = utils.Episode(environment=self.environment,\
                  agent=self.agent,\
                  model_name=self.model_name,\
                num_rounds=num_rounds,\
                state_interpreter=self.state_interpreter
                  )
          episode.run(verbose=4, learn=learn)
        print('Number of rounds:', len(episode.agent.rewards) - 1)
        print('Total reward:', np.nansum(episode.agent.rewards))
            
    def train(self):
        '''
        Trains agent.
        '''
        # Creates environment
        self.load_env(render_mode=None)
        # Creates episode
        episode = utils.Episode(environment=self.environment,\
                agent=self.agent,\
                model_name=self.model_name,\
                num_rounds=self.num_rounds,\
                state_interpreter=self.state_interpreter
                  )
        # Run simulation
        df = episode.simulate(num_episodes=self.num_episodes, file=self.file_csv)
        print(f'Data saved to {self.file_csv}')
        # Save agent to file
        self.save_agent()
        # Plot results
        p = utils.Plot(df)
        if self.num_episodes == 1:
          p.plot_round_reward(file=self.file_png)    
        else:
          p.plot_rewards(file=self.file_png) 
        print(f'Plot saved to {self.file_png}')
        # Save losses if agent uses NN
        if hasattr(self.agent.Q, 'losses'):
          losses = self.agent.Q.losses
          fig, ax = plt.subplots(figsize=(4,3.5))
          ax = sns.lineplot(x=range(len(losses)), y=losses)
          ax.set_xlabel("Epoch",fontsize=14)
          ax.set_ylabel("Loss",fontsize=14)
          plt.savefig(self.file_losses, dpi=300, bbox_inches="tight")

    def test(self, to_df=False, num_episodes=100):
        '''
        Test the trained agent.
        '''
        # Creates environment
        self.load_env(render_mode=None)
        # Creates episode
        episode = utils.Episode(environment=self.environment,\
                agent=self.agent,\
                model_name=self.model_name,\
                num_rounds=self.num_rounds,\
                state_interpreter=self.state_interpreter
                  )
        # Run simulation
        df = episode.simulate(num_episodes=num_episodes, learn=False)
        if to_df:
           # return dataframe
           self.data = df
        else:
          # Plot results
          p = utils.Plot(df)
          p.plot_histogram_rewards(self.file_test)
          print(f'Plot saved to {self.file_test}')

    def sweep(self, parameter:str, values:list, num_simulations:int=50):
        '''
        Runs a sweep over the specified parameter 
        with the specified values.
        '''
        # Creates environment
        self.load_env(render_mode=None)
        # Creates experiment
        experiment = utils.Experiment(environment=self.environment,\
                num_rounds=self.num_rounds,\
                num_episodes=self.num_episodes,\
                num_simulations=num_simulations,\
                state_interpreter=self.state_interpreter
                  )
        # Run sweep
        experiment.run_sweep1(agent=self.agent, \
                       name=self.model_name, \
                       parameter=parameter, \
                       values=values, \
                       measures=['reward'])

    def sweep2(self,\
              parameter1:str, \
              values1:list, \
              parameter2:str, \
              values2:list, 
              num_simulations:int=50):
        '''
        Runs a sweep over the two parameters 
        with the specified values.
        '''
        # Creates environment
        self.load_env(render_mode=None)
        # Creates experiment
        experiment = utils.Experiment(environment=self.environment,\
                num_rounds=self.num_rounds,\
                num_episodes=self.num_episodes,\
                num_simulations=num_simulations,\
                state_interpreter=self.state_interpreter
                  )
        # Run sweep
        experiment.run_sweep2(agent=self.agent, \
                       name=self.model_name, \
                       parameter1=parameter1, \
                       values1=values1,\
                       parameter2=parameter2, \
                       values2=values2
                       )

