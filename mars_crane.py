"""
MarsLander simulates a sky crane landing on Mars

Based on the Lunar Lander (https://github.com/fakemonk1/Reinforcement-Learning-Lunar_Lander)





Screencasting: 
ffmpeg -f x11grab -show_region 1 -r 25 -s 600x400 -i :1+75,55 -c:v libx264 mars_skycrane.mp4

"""

# import gym - not used any more, uses: from mars_lander_environment import MarsLander
import numpy as np
import pandas as pd
from collections import deque
import random
from pyglet.gl.gl import GL_SAMPLES

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import load_model

from mars_lander_environment import MarsLander

import pickle
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

class DQN:
    def __init__(self, env, lr, gamma, epsilon, epsilon_decay):

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.counter = 0

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.rewards_list = []

        self.replay_memory_buffer = deque(maxlen=500000)
        self.batch_size = 256 #64
        self.epsilon_min = 0.01
        self.num_action_space = self.action_space.n
        self.num_observation_space = env.observation_space.shape[0]
        self.model = self.initialize_model()

    def initialize_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.num_observation_space, activation='sigmoid'))
        model.add(Dense(256, activation=relu))
        model.add(Dense(128, activation=relu))
        model.add(Dense(self.num_action_space, activation='linear'))

        # Compile the model
        model.compile(loss=mean_squared_error,optimizer=Adam(lr=self.lr))
        print(model.summary())
        return model

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            act = random.randrange(self.num_action_space)
            # if act==4:
            #     if random.randrange(100)>2:
            #         act=0
            return act

        predicted_actions = self.model.predict(state)
        return np.argmax(predicted_actions[0])

    def add_to_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory_buffer.append((state, action, reward, next_state, done))

    def learn_and_update_weights_by_reply(self):

        # replay_memory_buffer size check
        if len(self.replay_memory_buffer) < self.batch_size/10 or self.counter != 0: # TODO early learning: start at batch_size//x ?
            return

        # Early Stopping # why that??????????????????????? TODO
        # if np.mean(self.rewards_list[-10:]) > 250:
        #     return
        samples = min(len(self.replay_memory_buffer),self.batch_size)
        random_sample = self.get_random_sample_from_replay_mem(samples=samples)
        states, actions, rewards, next_states, done_list = self.get_attribues_from_sample(random_sample)
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
        target_vec = self.model.predict_on_batch(states)
        indexes = np.array([i for i in range(samples)])
        target_vec[[indexes], [actions]] = targets # type: ignore

        self.model.fit(states, target_vec, epochs=1, verbose=0)

    def get_attribues_from_sample(self, random_sample):
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        return np.squeeze(states), actions, rewards, next_states, done_list

    def get_random_sample_from_replay_mem(self, samples):
        random_sample = random.sample(self.replay_memory_buffer, samples)
        return random_sample

    def train(self, num_episodes=2000, auto_stop_reward=250, checkpoint_intervall:int=False, start_episode:int=0):
        """Train the Deep Q Network

        Args:
            num_episodes (int, optional):   Maximum Number of Episodes. Defaults to 2000.
            auto_stop_reward (int, optional): learning stops if moving average of 
                last 100 rewards is this or above. Defaults to 250.
                Set to 0 or False to disable.
            checkpoint_intervall (int, optional): If > 0, the intervall in episodes to save model 
                checkpoints (model.h5) files. Defaults to False.
        """
        start_time = datetime.now()
        for episode in range(start_episode, num_episodes+start_episode):
            state = env.reset()
            reward_for_episode = 0
            num_steps = 1000
            state = np.reshape(state, [1, self.num_observation_space])
            for step in range(num_steps):
                env.render()
                received_action = self.get_action(state)
                # print("received_action:", received_action)
                next_state, reward, done, info = env.step(received_action)
                next_state = np.reshape(next_state, [1, self.num_observation_space])
                # Store the experience in replay memory
                self.add_to_replay_memory(state, received_action, reward, next_state, done)
                # add up rewards
                reward_for_episode += reward
                state = next_state
                self.update_counter()
                self.learn_and_update_weights_by_reply()

                if done:
                    break
            self.rewards_list.append(reward_for_episode)

            # Decay the epsilon after each experience completion
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Check for breaking condition
            last_rewards_mean = np.mean(self.rewards_list[-100:])
            if auto_stop_reward>0 and last_rewards_mean > auto_stop_reward:
                print("DQN Training Complete...")
                break
            print(f"{datetime.now()-start_time} Episode:{episode:5d} Reward:{reward_for_episode:10.3f} Average Reward:{last_rewards_mean:10.3f} epsilon: {self.epsilon:6.3%}")
            if checkpoint_intervall:
                if episode>0 and episode % checkpoint_intervall==0:
                    self.save(f"model3_skycrane_checkpoint_episode{episode:06d}.h5")

    def update_counter(self):
        self.counter += 1
        step_size = 5
        self.counter = self.counter % step_size

    def save(self, name):
        self.model.save(name)


def test_already_trained_model(trained_model, env):
    rewards_list = []
    num_test_episode = 100
    #env = gym.make("LunarLander-v2")
    print("Starting Testing of the trained model...")

    step_count = 1000

    for test_episode in range(num_test_episode):
        current_state = env.reset()
        num_observation_space = env.observation_space.shape[0]
        current_state = np.reshape(current_state, [1, num_observation_space])
        reward_for_episode = 0
        for step in range(step_count):
            env.render()
            selected_action = np.argmax(trained_model.predict(current_state)[0])
            new_state, reward, done, info = env.step(selected_action)
            new_state = np.reshape(new_state, [1, num_observation_space])
            current_state = new_state
            reward_for_episode += reward
            if done:
                break
        rewards_list.append(reward_for_episode)
        print(test_episode, "\t: Episode || Reward: ", reward_for_episode)

    return rewards_list


def plot_df(df, chart_name, title, x_axis_label, y_axis_label):
    plt.rcParams.update({'font.size': 17}) #type:ignore
    df['rolling_mean'] = df[df.columns[0]].rolling(100).mean()
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    # plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    # plt.ylim((-400, 300))
    fig = plot.get_figure()
    plt.legend().set_visible(False)
    fig.savefig(chart_name)


def plot_df2(df, chart_name, title, x_axis_label, y_axis_label):
    df['mean'] = df[df.columns[0]].mean()
    plt.rcParams.update({'font.size': 17}) # type:ignore
    plt.figure(figsize=(15, 8))
    plt.close()
    plt.figure()
    # plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
    plot = df.plot(linewidth=1.5, figsize=(15, 8))
    plot.set_xlabel(x_axis_label)
    plot.set_ylabel(y_axis_label)
    # plt.ylim((0, 300))
    # plt.xlim((0, 100))
    plt.legend().set_visible(False)
    fig = plot.get_figure()
    fig.savefig(chart_name)



if __name__ == '__main__':
    env = MarsLander(tether_action=True, render_reward_indicator=True) #gym.make('LunarLander-v2')

    # set seeds
    env.seed(21)
    np.random.seed(21)

    # setting up params
    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 3000
    start_episode = 0
    print('Start')
    model = DQN(env, lr, gamma, epsilon, epsilon_decay)

    restart_model_episode = None #600

    

    if not restart_model_episode is None:
        model.epsilon = epsilon * epsilon_decay**restart_model_episode
        training_episodes = 3000 - restart_model_episode
        model.model.load_weights(f"model_skycrane_checkpoint_episode{restart_model_episode:06d}.h5")
        start_episode = restart_model_episode

    model.train(
        num_episodes=training_episodes, 
        auto_stop_reward=250,
        checkpoint_intervall=50,
        start_episode=start_episode
        )

    # Save Everything
    save_dir = "final_model_"
    # Save trained model
    model.save(save_dir + "trained_model.h5")

    # Save Rewards list
    pickle.dump(model.rewards_list, open(save_dir + "train_rewards_list.p", "wb"))
    rewards_list = pickle.load(open(save_dir + "train_rewards_list.p", "rb"))

    # plot reward in graph
    reward_df = pd.DataFrame(rewards_list)
    plot_df(reward_df, "Figure 1- Reward for each training episode", "Reward for each training episode", "Episode","Reward")

    # Test the model
    trained_model = load_model(save_dir + "trained_model.h5")
    test_rewards = test_already_trained_model(trained_model, env)
    pickle.dump(test_rewards, open(save_dir + "test_rewards.p", "wb"))
    test_rewards = pickle.load(open(save_dir + "test_rewards.p", "rb"))

    plot_df2(pd.DataFrame(test_rewards), "Figure 2- Reward for each testing episode","Reward for each testing episode", "Episode", "Reward")
    print("Training and Testing Completed...!")
