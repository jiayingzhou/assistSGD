import copy

print('begin process')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
# from torch.utils.tensorboard import SummaryWriter
from collections import deque
import argparse
import time
from gym import wrappers
# import pickle
import os
# from pytorch_memlab import LineProfiler,MemReporter


parser = argparse.ArgumentParser()
# parser.add_argument('--path', help='path',default=None)
parser.add_argument('--record', help='record', default=False)
parser.add_argument('--reward_threshold', help='reward', default=200, type=float)
parser.add_argument('--assist_round', help='assistance_round', default=2, type=int)
# parser.add_argument('--epoch_train_round', help='number of such epoch rounds', default=1, type=int)
parser.add_argument('--iteration', help='iteration_round', default=999, type=int)
parser.add_argument('--device', help='GPU or CPU', default='cpu')
parser.add_argument('--setting', help='choose test setting', default=2, type=int)
parser.add_argument('--mean_loss', help='The way choose loss: mean or not?', default=False)
parser.add_argument('--fl_round', help='rounds in FLSGD', default=2, type=int)
parser.add_argument('--fl_epoch', help='epochs for training in FLSGD', default=3, type=int)

from gym.envs.box2d.lunar_landerV2 import LunarLander_new

args = parser.parse_args()
record = args.record
reward_threshold = args.reward_threshold
assistance_round = args.assist_round
iteration = args.iteration
# epoch_round = args.epoch_train_round
device = torch.device(args.device)
setting = args.setting
mean_loss = args.mean_loss
fl_epoch = args.fl_epoch
fl_round = args.fl_round

print('\n device is ', device)
print('\n record is ', record)
print('\n setting is ', setting)

subfile = 'setting'+str(setting)+'/ite'+str(iteration)+'/'
if not os.path.exists('arl_video/'+subfile):
    os.makedirs('arl_video/'+subfile)
if not os.path.exists('result/'+subfile):
    os.makedirs('result/'+subfile)
if not os.path.exists('result/'+subfile+'share/'):
    os.makedirs('result/'+subfile+'share/')

class Params:
    EPOCH_SIZE_IN_EACH_TRAJECTORY = 128  # How many epochs we want to pack when transferring parameters
    EPOCH_SIZE_IN_SHARE = 2    # How many parameters will be shared during each assistance
    ALPHA = 5e-3       # learning rate
    EPISODE_SIZE_IN_EACH_EPOCH = 64   # how many episodes we want to pack into an epoch
    GAMMA = 0.99        # discount rate
    HIDDEN_SIZE = 64    # number of hidden nodes we have in our dnn
    BETA = 0.1          # the entropy bonus multiplier
    reward_threshold = reward_threshold   # The stopping rule: when the last mean reward from 100 episodes > reward_threshold, stop!
    record = record   # whether to record the video or not

# Q-table is replaced by a neural network
class Agent(nn.Module):


    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Agent, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True)
        )

    def forward(self, x):
        x = normalize(x, dim=1)
        x = self.model(x)
        return x


class PolicyGradient:
    def __init__(self, environment: str = "CartPole", LEG_SPRING_TORQUE=40, map=None, msg1=None, msg2=None):

        self.ALPHA = Params.ALPHA
        self.EPISODE_SIZE_IN_EACH_EPOCH = Params.EPISODE_SIZE_IN_EACH_EPOCH
        self.GAMMA = Params.GAMMA
        self.HIDDEN_SIZE = Params.HIDDEN_SIZE
        self.BETA = Params.BETA
        self.EPOCH_SIZE_IN_EACH_TRAJECTORY = Params.EPOCH_SIZE_IN_EACH_TRAJECTORY
        self.EPOCH_SIZE_IN_SHARE = Params.EPOCH_SIZE_IN_SHARE
        self.reward_threshold = Params.reward_threshold
        # self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.DEVICE = device
        self.exit_flag = False
        self.msg1 = msg1
        self.msg2 = msg2
        self.record = record
        # create the environment
        self.env = environment
        self.LEG_SPRING_TORQUE = LEG_SPRING_TORQUE
        # self.height_level = height_level
        self.map= map
        # self.map_n = map_n
        self.episode_threshold = 10
        self.epoch_threshold = 0
        # self.epoch_round = epoch_round

        # the agent driven by a neural network architecture
        self.agent = Agent(observation_space_size=self.env.observation_space.shape[0],
                           action_space_size=self.env.action_space.n,
                           hidden_size=self.HIDDEN_SIZE).to(self.DEVICE)

        self.opt = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)
        # the total_rewards record mean rewards in the latest 100 episodes
        self.total_rewards_each_episode = deque([], maxlen=100)

        # self.discount_rewards_each_epoch = []
        self.history_loss_each_epoch = deque([], maxlen=1000)
        self.train_discount_reward_each_epoch = deque([], maxlen=1000)   # record mean discounted reward in each epoch
        self.train_discount_reward_in_assistance = deque([], maxlen=assistance_round)  # record mean discounted reward in each assistance, only record the last epoch
        self.train_loss_in_assist = deque([], maxlen=assistance_round)
        # test
        self.test_rewards1 = deque([], maxlen=1000)    # record test addtive reward in each testing episode
        self.test_discount_reward1 = deque([], maxlen=1000) # record test discount reward in each testing episode
        self.test_seed1 = deque([], maxlen=1000)
        self.test_height1 = deque([], maxlen=1000)

        self.test_rewards2 = deque([], maxlen=1000)  # record test addtive reward in each testing episode
        self.test_discount_reward2 = deque([], maxlen=1000)  # record test discount reward in each testing episode
        self.test_seed2 = deque([], maxlen=1000)
        self.test_height2 = deque([], maxlen=1000)

        self.test_rewards3 = deque([], maxlen=1000)  # record test addtive reward in each testing episode
        self.test_discount_reward3 = deque([], maxlen=1000)  # record test discount reward in each testing episode
        self.test_seed3 = deque([], maxlen=1000)
        self.test_height3 = deque([], maxlen=1000)

        self.test_rewards4 = deque([], maxlen=1000)  # record test addtive reward in each testing episode
        self.test_discount_reward4 = deque([], maxlen=1000)  # record test discount reward in each testing episode
        self.test_seed4 = deque([], maxlen=1000)
        self.test_height4 = deque([], maxlen=1000)

        self.total_model_in_share = deque([], maxlen=self.EPOCH_SIZE_IN_SHARE)
        # This is for selecting parameters, different parameters in different models
        self.total_model_in_share.append(self.agent)
        self.memory_epoch_mean = deque([], maxlen=int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY/self.EPOCH_SIZE_IN_SHARE))

        # flag to figure out if we have render a single episode current epoch
        self.finished_rendering_this_epoch = False
        # for federated learning
        self.test_rewards_fl = deque([], maxlen=fl_round)
        self.test_loss_fl = deque([], maxlen=fl_round)
        self.test_discount_rewards_fl = deque([], maxlen=fl_round)
        # self.train_rewards_fl = deque([], maxlen=fl_round)
        self.train_loss_fl = deque([], maxlen=fl_round)
        self.train_discount_rewards_fl = deque([], maxlen=fl_round)


    def test_on_environment(self, provider, test=True, fedrated=False, test_signal='t1'):
        epoch = 0
        # test_loss = []
        if self.record:
            provider.env = wrappers.Monitor(provider.env, './arl_video/' + subfile+ self.msg1 + '/' + self.msg2+'_on_' + provider.msg2, video_callable=lambda episode_id: True,
                                   force=True)
        # while epoch < self.EPOCH_SIZE_IN_EACH_TRAJECTORY:

        loss, mean_discount_rewards_epoch = provider.play_epoch_use_model_from_agent(self, test=test, test_signal=test_signal)


        # reset the rendering flag
        self.finished_rendering_this_epoch = False


            # test_loss.append(loss.data)
        if fedrated:
            self.test_loss_fl.append(loss)
            self.test_discount_rewards_fl.append(mean_discount_rewards_epoch)
        if self.record:
            self.env = self.env.unwrapped
        self.env.close()
        # return test_loss


    def get_assistance_from(self, provider, assist_round):
        """
                    The main interface for the Policy Gradient solver
                """


        # init the epoch arrays
        # used for entropy calculation
        self.memory_epoch_mean = deque([], maxlen=int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY/self.EPOCH_SIZE_IN_SHARE))
        self.total_model_in_share = deque([], maxlen=int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY/self.EPOCH_SIZE_IN_SHARE))

        # choose parameter who yield the smallest loss
        # We use all candidate parameters to perform episodes, and calculate loss (loss1 + loss2)
        # if len(provider.total_model_in_share) > 64:
        #     length = 50
        #     step_size = int(len(provider.total_model_in_share)/50)
        # else:
        #     length = len(provider.total_model_in_share)
        #     step_size = 1

        length = len(provider.total_model_in_share)
        step_size = 1

        # choose parameter from the provider first
        min_loss = np.Inf
        for param_ind in range(length):
            param_cand = param_ind * step_size
            provider.agent = provider.total_model_in_share[param_cand]
            loss, discount_reward_epoch_mean = self.play_epoch_use_model_from_agent(provider, record_reward=False)
            # take negative
            neg_discount_reward_epoch_mean = -(discount_reward_epoch_mean + provider.memory_epoch_mean[param_cand])
            if neg_discount_reward_epoch_mean < min_loss:
                max_ind = param_cand
                min_loss = neg_discount_reward_epoch_mean
            # decide the parameters and use it as starting point
        provider.agent = provider.total_model_in_share[max_ind]

        # init the episode and the epoch
        # init local training
        epoch = 0
        # loss_in_accumulate_epoch = deque([], maxlen=100)

        # launch a monitor at each round of assistance
        if self.record:
            self.env = wrappers.Monitor(self.env, './arl_video/' + subfile + self.msg1 + '/'+ self.msg2 + '/'+ str(assist_round), video_callable=lambda episode_id: True,
                                   force=True)

        while epoch < self.EPOCH_SIZE_IN_EACH_TRAJECTORY:
            if epoch == 0:
                loss, discount_reward_epoch_mean = self.play_epoch_use_model_from_agent(provider)
            else:
                loss, discount_reward_epoch_mean = self.play_epoch_use_model_from_agent(self)

            self.train_discount_reward_each_epoch.append(discount_reward_epoch_mean)

            # loss_in_accumulate_epoch.append(loss.data.cpu().numpy())
            # append loss and parameters for next round of assitance: user assist provider
            self.history_loss_each_epoch.append(loss.data)
            if epoch % self.EPOCH_SIZE_IN_SHARE == 0:
                self.memory_epoch_mean.append(discount_reward_epoch_mean)
                self.total_model_in_share.append(self.agent)
            # increment the epoch
            epoch += 1


            # reset the rendering flag
            self.finished_rendering_this_epoch = False
            # feedback
            print("\r",
                  f"Epoch: {epoch}, Avg Return rewards: {np.mean(self.total_rewards_each_episode):.3f}",
                  end="",
                  flush=True)

            # make epoch >=1, to make sure there is at least one gradient

            if np.mean(self.total_rewards_each_episode) > self.reward_threshold:
                print('\n solved!!')
                # record the discount_reward_epoch_mean in the end of each assistance
                self.train_discount_reward_in_assistance.append(discount_reward_epoch_mean)
                self.train_loss_in_assist.append(loss.data.cpu().numpy())
                if epoch == 1:  # no train, already converge
                    self.exit_flag = True
                break


            # zero the gradient
            self.opt.zero_grad()
            # backprop
            loss.backward()
            # update the parameters
            self.opt.step()


            if epoch == (self.EPOCH_SIZE_IN_EACH_TRAJECTORY - 1):
                # record the discount_reward_epoch_mean in the end of each assistance
                self.train_discount_reward_in_assistance.append(discount_reward_epoch_mean)
                self.train_loss_in_assist.append(loss.data.cpu().numpy())
            if self.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

            del loss, discount_reward_epoch_mean

            # if np.mean(self.total_rewards_each_episode) > 100:
            #     self.opt = optim.SGD(params=self.agent.parameters(), lr=0.2*self.ALPHA)


        # unwrap the monitor for next round of wrapping
        if self.record:
            self.env = self.env.unwrapped
        # self.env.close()

    # close the environment
    #     self.env.close()

        # close the writer
        # self.writer.close()

    def self_train(self, tic, fedrated=False):
        """
                    The main interface for the Policy Gradient solver
                """

        # choose parameter who yield the smallest loss
        # We use all candidate parameters to perform episodes, and calculate loss (loss1 + loss2)

        self.memory_epoch_mean = deque([], maxlen=int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY / self.EPOCH_SIZE_IN_SHARE))
        self.total_model_in_share = deque([], maxlen=int(self.EPOCH_SIZE_IN_EACH_TRAJECTORY / self.EPOCH_SIZE_IN_SHARE))

        epoch = 0

        if self.record:
            self.env = wrappers.Monitor(self.env, './arl_video/' + subfile + self.msg1 + '/'+ self.msg2, video_callable=lambda episode_id: True,
                                   force=True)
        while epoch < self.EPOCH_SIZE_IN_EACH_TRAJECTORY:
            loss, dicount_reward_epoch_mean = self.play_epoch_use_model_from_agent(self)

            # if the epoch is over - we have epoch trajectories to perform the policy gradient

            # reset the rendering flag
            self.finished_rendering_this_epoch = False
            # append loss and parameters for next round of assitance: user assist provider
            self.history_loss_each_epoch.append(loss.data)
            self.train_discount_reward_each_epoch.append(dicount_reward_epoch_mean)

            if epoch % self.EPOCH_SIZE_IN_SHARE == 0:
                self.memory_epoch_mean.append(dicount_reward_epoch_mean)
                self.total_model_in_share.append(self.agent)
                toc = time.time()
                print('\n The running minute is ', (toc - tic) / 60)
            # increment the epoch
            epoch += 1

            # feedback
            print("\r", f"Epoch: {epoch}, Avg Return rewards: {np.mean(self.total_rewards_each_episode):.3f}",
                  end="",
                  flush=True)


            if np.mean(self.total_rewards_each_episode) > self.reward_threshold:
                print('\n solved!!')
                # record the discount_reward_epoch_mean in the end of each fl
                if fedrated:
                    self.train_discount_rewards_fl.append(dicount_reward_epoch_mean)
                    self.train_loss_fl.append(loss)
                    if epoch == 1:   # no train, already converge
                        self.exit_flag = True
                break
            if fedrated:
                if epoch == (self.EPOCH_SIZE_IN_EACH_TRAJECTORY):
                    # record the discount_reward_epoch_mean in the end of each fl
                    self.train_discount_rewards_fl.append(dicount_reward_epoch_mean)
                    self.train_loss_fl.append(loss)

            if self.DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

            # zero the gradient
            self.opt.zero_grad()
            # backprop
            loss.backward()
            # update the parameters
            self.opt.step()

            del loss, dicount_reward_epoch_mean


        if self.record:
            self.env = self.env.unwrapped
    # close the environment
    #     self.env.close()

        # close the writer
        # self.writer.close()

    def play_episode_use_agent(self, provider, standardize=True):
        """
            Plays an episode of the environment.
            episode: the episode counter
            Returns:
                sum_weighted_log_probs: the sum of the log-prob of an action multiplied by the reward-to-go from that state
                episode_logits: the logits of every step of the episode - needed to compute entropy for entropy bonus
                finished_rendering_this_epoch: pass-through rendering flag
                sum_of_rewards: sum of the rewards for the episode - needed for the average over 200 episode statistic
        """
        # reset the environment to a random initial state every epoch
        ll = len(self.map.h)
        index = torch.randperm(ll)[0]
        seed = self.map.seed[index]
        height = self.map.h[index]
        self.env.seed(int(seed))

        state = self.env.reset(height_level=height, LEG_SPRING_TORQUE=self.LEG_SPRING_TORQUE)

        # initialize the episode arrays
        episode_actions = torch.empty(size=(0,), dtype=torch.long, device=self.DEVICE)
        episode_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        cumulative_average_rewards = np.empty(shape=(0,), dtype=float)
        # cumulative_sd_rewards = np.empty(shape=(0,), dtype=float)
        episode_rewards = np.empty(shape=(0,), dtype=float)

        # episode loopf
        while True:

            # render the environment for the first episode in the epoch
            # if not self.finished_rendering_this_epoch:
            #     self.env.render()

            # get the action logits from the agent - (preferences)
            action_logits = provider.agent(torch.tensor(state).float().unsqueeze(dim=0).to(self.DEVICE))

            # append the logits to the episode logits list
            episode_logits = torch.cat((episode_logits, action_logits), dim=0)

            # sample an action according to the action distribution
            action = Categorical(logits=action_logits).sample()

            # append the action to the episode action list to obtain the trajectory
            # we need to store the actions and logits so we could calculate the gradient of the performance
            episode_actions = torch.cat((episode_actions, action), dim=0)

            # take the chosen action, observe the reward and the next state

            if self.DEVICE.type == 'cpu':
                state, reward, done, _ = self.env.step(action=action.cpu().item())
            else:
                state, reward, done, _ = self.env.step(action=action.cuda().item())
            # append the reward to the rewards pool that we collect during the episode
            # we need the rewards so we can calculate the weights for the policy gradient
            # and the baseline of average
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)

            # here the average reward is state specific
            cumulative_average_rewards = np.concatenate((cumulative_average_rewards,
                                              np.expand_dims(np.mean(episode_rewards), axis=0)),
                                             axis=0)
            # cumulative_sd_rewards = np.concatenate((cumulative_sd_rewards,
            #                                              np.expand_dims(np.std(episode_rewards), axis=0)),
            #                                             axis=0)

            if episode_logits.shape[0] > 1000:   # done if the running steps are too much
                done = True

            episode_rewards_sum = sum(episode_rewards)
            if episode_rewards_sum < -250:
                done = True

            # the episode is over
            if done:

                # increment the episode


                # turn the rewards we accumulated during the episode into the rewards-to-go:
                # earlier actions are responsible for more rewards than the later taken actions
                discounted_rewards_to_go = PolicyGradient.get_discounted_rewards(rewards=episode_rewards,
                                                                                 gamma=self.GAMMA)
                discounted_tot_rewards = discounted_rewards_to_go[0]
                if standardize:
                    discounted_rewards_to_go -= cumulative_average_rewards  # baseline - state specific average
                    # discounted_tot_rewards /= cumulative_sd_rewards
                # # calculate the sum of the rewards for the running average metric
                sum_of_rewards = np.sum(episode_rewards)

                # set the mask for the actions taken in the episode
                mask = one_hot(episode_actions, num_classes=self.env.action_space.n)

                # calculate the log-probabilities of the taken actions
                # mask is needed to filter out log-probabilities of not related logits
                episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

                # weight the episode log-probabilities by the rewards-to-go
                episode_weighted_log_probs = episode_log_probs * \
                    torch.tensor(discounted_rewards_to_go).float().to(self.DEVICE)

                # calculate the sum over trajectory of the weighted log-probabilities
                if not mean_loss:
                    sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)
                else:
                    sum_weighted_log_probs = torch.mean(episode_weighted_log_probs).unsqueeze(dim=0)

                # won't render again this epoch
                self.finished_rendering_this_epoch = True
                # env_wrap.close()
                self.env.close()


                return sum_weighted_log_probs, episode_logits, sum_of_rewards,discounted_tot_rewards, seed, height

    def calculate_loss(self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
            Calculates the policy "loss" and the entropy bonus
            Args:
                epoch_logits: logits of the policy network we have collected over the epoch
                weighted_log_probs: loP * W of the actions taken
            Returns:
                policy loss + the entropy bonus
                entropy: needed for logging
        """
        policy_loss = -1 * torch.mean(weighted_log_probs)

        # add the entropy bonus

        p = softmax(epoch_logits, dim=1)
        log_p = log_softmax(epoch_logits, dim=1)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
        if not mean_loss:
            entropy_bonus = -1 * self.BETA * entropy
        else:
            entropy_bonus = 0

        return policy_loss + entropy_bonus, entropy

    @staticmethod
    def get_discounted_rewards(rewards: np.array, gamma: float) -> np.array:
        """
            Calculates the sequence of discounted rewards-to-go.
            Args:
                rewards: the sequence of observed rewards
                gamma: the discount factor
            Returns:
                discounted_rewards: the sequence of the rewards-to-go
        """
        discounted_rewards = np.empty_like(rewards, dtype=float)
        for i in range(rewards.shape[0]):
            gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
            discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
            discounted_reward = np.sum(rewards[i:] * discounted_gammas)
            discounted_rewards[i] = discounted_reward
        return discounted_rewards

    def play_epoch_use_model_from_agent(self, provider, test=False, record_reward=True, test_signal=None): # record_reward = False, when receiving paras, no record


        epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)
        episode = 0
        discount_rewards_epoch = np.empty(shape=(0,), dtype=float)

        if test:
            episode_size = 4 * self.EPISODE_SIZE_IN_EACH_EPOCH
        else:
            episode_size = self.EPISODE_SIZE_IN_EACH_EPOCH
        # discounted_tot_rewards_epoch = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)
        # sum_of_episode_rewards_epoch = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)
        discounted_tot_rewards_epoch = deque([], maxlen=episode_size)
        sum_of_episode_rewards_epoch = deque([], maxlen=episode_size)
        seed_epoch = deque([], maxlen=episode_size)
        height_epoch = deque([], maxlen=episode_size)
        while episode < episode_size:
            (episode_weighted_log_prob_trajectory,
             episode_logits,
             sum_of_episode_rewards,
             discounted_tot_rewards, seed, height
             ) = self.play_episode_use_agent(provider)

            episode += 1

            # after each episode append the sum of total rewards to the deque
            if record_reward:
                if test:
                    # sum_of_episode_rewards_epoch = np.concatenate((sum_of_episode_rewards_epoch, np.array([sum_of_episode_rewards.cpu().numpy()])), axis=0)
                    # discounted_tot_rewards_epoch = np.concatenate((discounted_tot_rewards_epoch, np.array([discounted_tot_rewards.cpu().numpy()])), axis=0)
                    sum_of_episode_rewards_epoch.append(sum_of_episode_rewards)
                    discounted_tot_rewards_epoch.append(discounted_tot_rewards)
                    seed_epoch.append(seed)
                    height_epoch.append(height)
                else:
                    self.total_rewards_each_episode.append(sum_of_episode_rewards)


            # append the weighted log-probabilities of actions
            epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),
                                                 dim=0)

            # append the logits - needed for the entropy bonus calculation
            epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)
            discount_rewards_epoch = np.concatenate((discount_rewards_epoch, np.array([discounted_tot_rewards])), axis=0)

            # # convergence rule
            # if episode > 64:
            #     # print('standard err is ',np.std(rewards_epoch)/np.sqrt(episode-1))
            #     if np.std(rewards_epoch)/np.sqrt(episode-1) < self.episode_threshold:
            #         break

        # reset the rendering flag
        self.finished_rendering_this_epoch = False
        if test:
            if test_signal == 't1':
                provider.test_rewards1.append(sum_of_episode_rewards_epoch)
                provider.test_discount_reward1.append(discounted_tot_rewards_epoch)
                provider.test_height1.append(height_epoch)
                provider.test_seed1.append(seed_epoch)
            if test_signal == 't2':
                provider.test_rewards2.append(sum_of_episode_rewards_epoch)
                provider.test_discount_reward2.append(discounted_tot_rewards_epoch)
                provider.test_height2.append(height_epoch)
                provider.test_seed2.append(seed_epoch)
            if test_signal == 't3':
                provider.test_rewards3.append(sum_of_episode_rewards_epoch)
                provider.test_discount_reward3.append(discounted_tot_rewards_epoch)
                provider.test_height3.append(height_epoch)
                provider.test_seed3.append(seed_epoch)
            if test_signal == 't4':
                provider.test_rewards4.append(sum_of_episode_rewards_epoch)
                provider.test_discount_reward4.append(discounted_tot_rewards_epoch)
                provider.test_height4.append(height_epoch)
                provider.test_seed4.append(seed_epoch)
        loss, _ = self.calculate_loss(epoch_logits=epoch_logits,
                                            weighted_log_probs=epoch_weighted_log_probs)
        # if the epoch is over - we have epoch trajectories to perform the policy gradient
        return loss, np.mean(discount_rewards_epoch)






def train_assist(user_map, provider_map, single_agent, test_agent1,test_agent2,test_agent3,test_agent4, message=None):
    env = LunarLander_new()
    main_agent = PolicyGradient(environment=env, LEG_SPRING_TORQUE=40,map=user_map, msg1=message, msg2='u')
    assist_agent = PolicyGradient(environment=env, LEG_SPRING_TORQUE=40, map=provider_map,msg1=message,msg2='p')
    communication_round = assistance_round

    # deep copy the model from single agent
    main_agent.total_model_in_share = copy.deepcopy(single_agent.total_model_in_share)
    main_agent.memory_epoch_mean = copy.deepcopy(single_agent.memory_epoch_mean)



    for communication in range(communication_round):
        print("\n This is Assistance ", communication)

        tic = time.time()
        print("\n provider get assistance from user")
        assist_agent.get_assistance_from(main_agent, assist_round=communication)
        if assist_agent.exit_flag:
            break
        print('\n running time ', (time.time()-tic)/60)

        print("\n user get assistance from provider")
        tic = time.time()
        main_agent.get_assistance_from(assist_agent, assist_round=communication)

        main_agent.test_on_environment(test_agent1, test_signal='t1')
        main_agent.test_on_environment(test_agent2, test_signal='t2')
        main_agent.test_on_environment(test_agent3, test_signal='t3')
        main_agent.test_on_environment(test_agent4, test_signal='t4')
        print('\n ----------   Assistance ', communication, '  Feedback-------------------')
        print('\n The training loss at assistance', main_agent.train_loss_in_assist)
        print('\n The training discount reward at this assistance',
              main_agent.train_discount_reward_in_assistance[communication])

        print('\n main test reward mean on test 1: ')
        for iii in range(communication + 1):
            print(np.mean(main_agent.test_rewards1[iii]), ',')
        print('\n main test reward mean on test 2: ')
        for iii in range(communication + 1):
            print(np.mean(main_agent.test_rewards2[iii]), ',')
        print('\n main test reward mean on test 3: ')
        for iii in range(communication + 1):
            print(np.mean(main_agent.test_rewards3[iii]), ',')
        print('\n main test reward mean on test 4: ')
        for iii in range(communication + 1):
            print(np.mean(main_agent.test_rewards4[iii]), ',')

        print('\n ----------------------------------------------------------------')
        if main_agent.exit_flag:
            break
        torch.save({
            'u_history_loss': main_agent.history_loss_each_epoch,
            'u_model_state_dict': main_agent.agent.state_dict(),
            'u_optimizer_state_dict': main_agent.opt.state_dict(),
            'u_test_rewards1': main_agent.test_rewards1,
            'u_test_seed1': main_agent.test_seed1,
            'u_test_height1': main_agent.test_height1,
            'u_test_rewards2': main_agent.test_rewards2,
            'u_test_seed2': main_agent.test_seed2,
            'u_test_height2': main_agent.test_height2,
            'u_test_rewards3': main_agent.test_rewards3,
            'u_test_seed3': main_agent.test_seed3,
            'u_test_height3': main_agent.test_height3,
            'u_test_rewards4': main_agent.test_rewards4,
            'u_test_seed4': main_agent.test_seed4,
            'u_test_height4': main_agent.test_height4,
            'u_test_discount_reward1': main_agent.test_discount_reward1,
            'u_test_discount_reward2': main_agent.test_discount_reward2,
            'u_test_discount_reward3': main_agent.test_discount_reward3,
            'u_test_discount_reward4': main_agent.test_discount_reward4,
            'u_episode_reward': main_agent.total_rewards_each_episode,
            'u_discount_reward_in_assist': main_agent.train_discount_reward_in_assistance,
            'p_history_loss': assist_agent.history_loss_each_epoch,
            'p_model_state_dict': assist_agent.agent.state_dict(),
            'p_optimizer_state_dict': assist_agent.opt.state_dict(),
            'p_test_rewards1': assist_agent.test_rewards1,
            'p_test_seed1': assist_agent.test_seed1,
            'p_test_height1': assist_agent.test_height1,
            'p_test_rewards2': assist_agent.test_rewards2,
            'p_test_seed2': assist_agent.test_seed2,
            'p_test_height2': assist_agent.test_height2,
            'p_test_rewards3': assist_agent.test_rewards3,
            'p_test_seed3': assist_agent.test_seed3,
            'p_test_height3': assist_agent.test_height3,
            'p_test_rewards4': assist_agent.test_rewards4,
            'p_test_seed4': assist_agent.test_seed4,
            'p_test_height4': assist_agent.test_height4,
            'p_test_discount_reward1': assist_agent.test_discount_reward1,
            'p_test_discount_reward2': assist_agent.test_discount_reward2,
            'p_test_discount_reward3': assist_agent.test_discount_reward3,
            'p_test_discount_reward4': assist_agent.test_discount_reward4,
            'p_episode_reward': assist_agent.total_rewards_each_episode,
            'p_discount_reward_in_assist': assist_agent.train_discount_reward_in_assistance,
        }, 'result/' + subfile + 'assist'+ str(communication))
        print('\n running time ', (time.time() - tic) / 60)

        # save shared parameter and model for future training if not converged， save everytime
        for ii in range(len(main_agent.total_model_in_share)):
            torch.save({
                'u_share_loss': main_agent.memory_epoch_mean[ii],
                'u_share_model': main_agent.total_model_in_share[ii].state_dict(),
            }, 'result/' + subfile + 'share/assist'+str(communication)+'u'+str(ii))
        for jj in range(len(assist_agent.total_model_in_share)):
            torch.save({
                'p_share_loss': assist_agent.memory_epoch_mean[jj],
                'p_share_model': assist_agent.total_model_in_share[jj].state_dict(),
            }, 'result/' + subfile + 'share/assist'+str(communication)+'p'+str(jj))


    return main_agent, assist_agent

def single_train_and_test(single_agent, test_agent1,test_agent2, test_agent3,test_agent4, message=None,PRINT=True):
    # self train first
    tic = time.time()
    single_agent.self_train(tic=tic)
    single_agent.test_on_environment(test_agent1, test_signal='t1')
    single_agent.test_on_environment(test_agent2, test_signal='t2')
    single_agent.test_on_environment(test_agent3, test_signal='t3')
    single_agent.test_on_environment(test_agent4, test_signal='t4')
    if PRINT:
        print('\n ---------- ', message ,' agent  Feedback-------------------')
        print('\n The training loss', single_agent.history_loss_each_epoch[-1])
        print('\n The training discount reward',
              single_agent.train_discount_reward_each_epoch[-1])

        print('\n main test reward mean on test 1: ')
        print(np.mean(single_agent.test_rewards1[0]), ',')
        print('\n main test reward mean on test 2: ')
        print(np.mean(single_agent.test_rewards2[0]), ',')
        print('\n main test reward mean on test 3: ')
        print(np.mean(single_agent.test_rewards3[0]), ',')
        print('\n main test reward mean on test 4: ')
        print(np.mean(single_agent.test_rewards4[0]), ',')
        print('---------------------------------------------------------')
    torch.save({
        'history_loss': single_agent.history_loss_each_epoch,
        'model_state_dict': single_agent.agent.state_dict(),
        'optimizer_state_dict': single_agent.opt.state_dict(),
        'epoch_disc_reward': single_agent.train_discount_reward_each_epoch,
        'test_rewards1': single_agent.test_rewards1,
        'test_rewards2': single_agent.test_rewards2,
        'test_rewards3': single_agent.test_rewards3,
        'test_rewards4': single_agent.test_rewards4,
        'test_seed1': single_agent.test_seed1,
        'test_height1': single_agent.test_height1,
        'test_discount_reward1': single_agent.test_discount_reward1,
        'test_seed2': single_agent.test_seed2,
        'test_height2': single_agent.test_height2,
        'test_discount_reward2': single_agent.test_discount_reward2,
        'test_seed3': single_agent.test_seed3,
        'test_height3': single_agent.test_height3,
        'test_discount_reward3': single_agent.test_discount_reward3,
        'test_seed4': single_agent.test_seed4,
        'test_height4': single_agent.test_height4,
        'test_discount_reward4': single_agent.test_discount_reward4,
        'episode_reward': single_agent.total_rewards_each_episode,
    }, 'result/' + subfile + message)

def train_single(user_map, message=None):

    env = LunarLander_new()


    main_agent = PolicyGradient(environment=env, LEG_SPRING_TORQUE=40, map=user_map, msg1=message, msg2='u')

    tic = time.time()

    main_agent.self_train(tic=tic)



    torch.save({
        'history_loss': main_agent.history_loss_each_epoch,
        'model_state_dict': main_agent.agent.state_dict(),
        'optimizer_state_dict': main_agent.opt.state_dict(),
        'epoch_disc_reward': main_agent.train_discount_reward_each_epoch,
        'test_rewards1': main_agent.test_rewards1,
        'test_rewards2': main_agent.test_rewards2,
        'test_rewards3': main_agent.test_rewards3,
        'test_rewards4': main_agent.test_rewards4,
        'test_seed1': main_agent.test_seed1,
        'test_height1': main_agent.test_height1,
        'test_loss1': main_agent.test_discount_reward1,
        'test_seed2': main_agent.test_seed2,
        'test_height2': main_agent.test_height2,
        'test_loss2': main_agent.test_discount_reward2,
        'test_seed3': main_agent.test_seed3,
        'test_height3': main_agent.test_height3,
        'test_loss3': main_agent.test_discount_reward3,
        'test_seed4': main_agent.test_seed4,
        'test_height4': main_agent.test_height4,
        'test_loss4': main_agent.test_discount_reward4,
    }, 'result/'+subfile+'oracle')

    return main_agent

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def train_fl(u_map, p_map, round, epoch_num, message, test_agent1, test_agent2, test_agent3, test_agent4):

    main_agent = PolicyGradient(environment=env, LEG_SPRING_TORQUE=40, map=u_map, msg1=message, msg2='u')
    main_agent.EPOCH_SIZE_IN_EACH_TRAJECTORY = epoch_num
    assist_agent = PolicyGradient(environment=env, LEG_SPRING_TORQUE=40, map=p_map, msg1=message, msg2='u')
    assist_agent.EPOCH_SIZE_IN_EACH_TRAJECTORY = epoch_num
    for rr in range(round):
        print('\n =================     FedSGD     ================')
        print('\n =========   This is round ', rr)
        global_w_list = []
        tic = time.time()
        main_agent.self_train(tic=tic, fedrated=True)
        if main_agent.exit_flag:
            break
        tic = time.time()
        assist_agent.self_train(tic=tic, fedrated=True)
        # average weights
        global_w_list.append(main_agent.agent.state_dict())
        global_w_list.append(assist_agent.agent.state_dict())
        global_weight = average_weights(global_w_list)
        main_agent.agent.load_state_dict(global_weight)
        assist_agent.agent.load_state_dict(global_weight)
        # test
        main_agent.test_on_environment(test_agent1, fedrated=True, test_signal='t1')
        main_agent.test_on_environment(test_agent2, fedrated=True, test_signal='t2')
        main_agent.test_on_environment(test_agent3, fedrated=True, test_signal='t3')
        main_agent.test_on_environment(test_agent4, fedrated=True, test_signal='t4')

        print('\n --------------   Round ', rr, '  performance   ---------------')
        print('\n The federated training loss', main_agent.train_loss_fl[-1])
        print('\n The federated training discount reward', main_agent.train_discount_rewards_fl[-1])

        print('\n main test reward mean on test 1: ')
        for iii in range(rr+1):
            print(np.mean(main_agent.test_rewards1[iii]), ',')
        print('\n main test reward mean on test 2: ')
        for iii in range(rr+1):
            print(np.mean(main_agent.test_rewards2[iii]), ',')
        print('\n main test reward mean on test 3: ')
        for iii in range(rr+1):
            print(np.mean(main_agent.test_rewards3[iii]), ',')
        print('\n main test reward mean on test 4: ')
        for iii in range(rr+1):
            print(np.mean(main_agent.test_rewards4[iii]), ',')

        print('\n --------------------------------------------------------------')


        torch.save({
            'fl_round': rr,
            'history_loss': main_agent.history_loss_each_epoch,
            'model_state_dict': main_agent.agent.state_dict(),
            'optimizer_state_dict': main_agent.opt.state_dict(),
            'epoch_disc_reward': main_agent.train_discount_reward_each_epoch,
            'train_discount_reward_fl': main_agent.train_discount_rewards_fl,
            'train_loss_fl': main_agent.train_loss_fl,
            'test_rewards1': main_agent.test_rewards1,
            'test_seed1': main_agent.test_seed1,
            'test_height1': main_agent.test_height1,
            'test_loss1': main_agent.test_discount_reward1,
            'test_rewards2': main_agent.test_rewards2,
            'test_seed2': main_agent.test_seed2,
            'test_height2': main_agent.test_height2,
            'test_loss2': main_agent.test_discount_reward2,
            'test_rewards3': main_agent.test_rewards3,
            'test_seed3': main_agent.test_seed3,
            'test_height3': main_agent.test_height3,
            'test_loss3': main_agent.test_discount_reward3,
            'test_rewards4': main_agent.test_rewards4,
            'test_seed4': main_agent.test_seed4,
            'test_height4': main_agent.test_height4,
            'test_loss4': main_agent.test_discount_reward4,
        }, 'result/' + subfile + 'fl')

    return main_agent, assist_agent



def map_list(map_uu, map_pp, map_tt1,map_tt2,map_tt3,map_tt4, ite):

    l1 = len(map_uu.steep_level)
    q1 = sum(map_uu.quantity)
    map_u = deque([], maxlen=q1)
    for i in range(l1):
        map_u += [map_uu.steep_level[i]]*map_uu.quantity[i]

    l2 = len(map_pp.steep_level)
    q2 = sum(map_pp.quantity)
    map_p = deque([], maxlen=q2)
    for j in range(l2):
        map_p += [map_pp.steep_level[j]] * map_pp.quantity[j]
    map_all = map_u + map_p

    lt1 = len(map_tt1.steep_level)
    qt1 = sum(map_tt1.quantity)
    map_t1 = deque([], maxlen=qt1)
    for k in range(lt1):
        map_t1 += [map_tt1.steep_level[k]] * map_tt1.quantity[k]

    lt2 = len(map_tt2.steep_level)
    qt2 = sum(map_tt2.quantity)
    map_t2 = deque([], maxlen=qt2)
    for k in range(lt2):
        map_t2 += [map_tt2.steep_level[k]] * map_tt2.quantity[k]

    lt3 = len(map_tt3.steep_level)
    qt3 = sum(map_tt3.quantity)
    map_t3 = deque([], maxlen=qt3)
    for k in range(lt3):
        map_t3 += [map_tt3.steep_level[k]] * map_tt3.quantity[k]

    lt4 = len(map_tt4.steep_level)
    qt4 = sum(map_tt4.quantity)
    map_t4 = deque([], maxlen=qt4)
    for k in range(lt4):
        map_t4 += [map_tt4.steep_level[k]] * map_tt4.quantity[k]

    class u:
        h = map_u
        seed = list(np.arange(q1)+ite*(q1+q2+qt1+qt2+qt3+qt4))
    class p:
        h = map_p
        seed = list(np.arange(q1, q1+q2)+ite*(q1+q2+qt1+qt2+qt3+qt4))
    class t1:
        h = map_t1
        seed = list(np.arange(q1+q2, q1+q2+qt1)+ite*(q1+q2+qt1+qt2+qt3+qt4))
    class t2:
        h = map_t2
        seed = list(np.arange(q1+q2+qt1, q1+q2+qt1+qt2)+ite*(q1+q2+qt1+qt2+qt3+qt4))
    class t3:
        h = map_t3
        seed = list(np.arange(q1+q2+qt1+qt2, q1+q2+qt1+qt2+qt3)+ite*(q1+q2+qt1+qt2+qt3+qt4))
    class t4:
        h = map_t4
        seed = list(np.arange(q1+q2+qt1+qt2+qt3, q1+q2+qt1+qt2+qt3+qt4)+ite*(q1+q2+qt1+qt2+qt3+qt4))
    class o:
        h = map_all
        seed = list(np.arange(q1+q2)+ite*(q1+q2+qt1+qt2+qt3+qt4))
    return u, p, o, t1, t2, t3, t4


def save_all_result(main_agent, assist_agent,oracle_agent, filename):
    torch.save({
        'u_history_loss': main_agent.history_loss_each_epoch,
        'u_model_state_dict': main_agent.agent.state_dict(),
        'u_optimizer_state_dict': main_agent.opt.state_dict(),
        'u_test_rewards1': main_agent.test_rewards1,
        'u_test_discount_reward1': main_agent.test_discount_reward1,
        'u_test_rewards2': main_agent.test_rewards2,
        'u_test_discount_reward2': main_agent.test_discount_reward2,
        'u_test_rewards3': main_agent.test_rewards3,
        'u_test_discount_reward3': main_agent.test_discount_reward3,
        'u_test_rewards4': main_agent.test_rewards4,
        'u_test_discount_reward4': main_agent.test_discount_reward4,
        # 'u_epoch_reward': main_agent.discount_rewards_each_epoch,
        'u_episode_reward': main_agent.total_rewards_each_episode,
        'u_epoch_disc_reward': main_agent.train_discount_reward_each_epoch,
        'p_history_loss': assist_agent.history_loss_each_epoch,
        'p_model_state_dict': assist_agent.agent.state_dict(),
        'p_optimizer_state_dict': assist_agent.opt.state_dict(),
        'p_test_rewards1': assist_agent.test_rewards1,
        'p_test_discount_reward1': assist_agent.test_discount_reward1,
        'p_test_rewards2': assist_agent.test_rewards2,
        'p_test_discount_reward2': assist_agent.test_discount_reward2,
        'p_test_rewards3': assist_agent.test_rewards3,
        'p_test_discount_reward3': assist_agent.test_discount_reward3,
        'p_test_rewards4': assist_agent.test_rewards4,
        'p_test_discount_reward4': assist_agent.test_discount_reward4,
        # 'p_epoch_reward': assist_agent.discount_rewards_each_epoch,
        'p_episode_reward': assist_agent.total_rewards_each_episode,
        'p_epoch_disc_reward': assist_agent.train_discount_reward_each_epoch,
        'o_history_loss': oracle_agent.history_loss_each_epoch,
        'o_model_state_dict': oracle_agent.agent.state_dict(),
        'o_optimizer_state_dict': oracle_agent.opt.state_dict(),
        'o_test_rewards1': oracle_agent.test_rewards1,
        'o_test_discount_reward1': oracle_agent.test_discount_reward1,
        'o_test_rewards2': oracle_agent.test_rewards2,
        'o_test_discount_reward2': oracle_agent.test_discount_reward2,
        'o_test_rewards3': oracle_agent.test_rewards3,
        'o_test_discount_reward3': oracle_agent.test_discount_reward3,
        'o_test_rewards4': oracle_agent.test_rewards4,
        'o_test_discount_reward4': oracle_agent.test_discount_reward4,
        # 'o_epoch_reward': oracle_agent.discount_rewards_each_epoch,
        'o_episode_reward': oracle_agent.total_rewards_each_episode,
        'o_epoch_disc_reward': oracle_agent.train_discount_reward_each_epoch,
    }, 'result/' + subfile + filename)

if __name__ == "__main__":
    if setting == 0:
        class map_user:
            steep_level = [0.2, 0.6]
            quantity = [10, 2]
        class map_provider:
            steep_level = [0.2, 0.4, 0.6, 0.8]
            quantity = [25, 25, 25, 25]
        class map_test1:
            steep_level = [0.2, 0.6]
            quantity = [1000, 200]

    if setting == 1:
        class map_user:
            steep_level = [0.2, 0.3]
            quantity = [5, 5]

        class map_provider:
            steep_level = [0.6, 0.7]
            quantity = [5, 5]

        class map_test1:
            steep_level = [0.2, 0.3]
            quantity = [500, 500]

    if setting == 2:
        class map_user:
            steep_level = [0.2, 0.6]
            quantity = [10, 2]
        class map_provider:
            steep_level = [0.2, 0.6]
            quantity = [100, 100]
        class map_test1:
            steep_level = [0.2, 0.6]
            quantity = [1000, 200]


    if setting == 3:
        class map_user:
            steep_level = [0.2, 0.6]
            quantity = [10, 2]
        class map_provider:
            steep_level = [0.2, 0.6]
            quantity = [10, 2]
        class map_test1:
            steep_level = [0.2, 0.6]
            quantity = [1000, 200]
    class map_test2:
        steep_level = list(np.random.uniform(0, 1, 1000))
        quantity = [1] * 1000
    class map_test3:
        steep_level = list(np.random.beta(5, 1, 1000))
        quantity = [1] * 1000
    class map_test4:
        steep_level = list(np.random.beta(1, 5, 1000))
        quantity = [1] * 1000


    env = LunarLander_new()

    u, p, o, t1, t2, t3, t4 = map_list(map_user, map_provider, map_test1, map_test2, map_test3, map_test4, iteration)
    start = time.time()
    # test part
    test_agent1 = PolicyGradient(environment=env, LEG_SPRING_TORQUE=40,
                                map=t1, msg1=None, msg2='t1')
    test_agent2 = PolicyGradient(environment=env, LEG_SPRING_TORQUE=40,
                                 map=t2, msg1=None, msg2='t2')
    test_agent3 = PolicyGradient(environment=env, LEG_SPRING_TORQUE=40,
                                 map=t3, msg1=None, msg2='t3')
    test_agent4 = PolicyGradient(environment=env, LEG_SPRING_TORQUE=40,
                                 map=t4, msg1=None, msg2='t4')
    # train first

    print('\n ==============     Start trainig single agent  ==============')
    single_agent = PolicyGradient(environment=env, LEG_SPRING_TORQUE=40, map=u, msg1='_single_', msg2='u')
    single_train_and_test(single_agent, test_agent1, test_agent2, test_agent3, test_agent4, message='single',
                          PRINT=True)
    main_agent, assist_agent = train_assist(u, p, single_agent,test_agent1, test_agent2, test_agent3, test_agent4, message='_assist_')

    oracle_agent = PolicyGradient(environment=env, LEG_SPRING_TORQUE=40, map=o, msg1='_oracle_', msg2='u')
    print('\n =============    Start training oracle agent   ==============')
    single_train_and_test(oracle_agent, test_agent1, test_agent2, test_agent3, test_agent4, message='oracle', PRINT=True)
    print('\n =============================================================')

    print('\n =================      Start FedSGD     ================')

    user1, user2 = train_fl(u, p, round=fl_round, epoch_num=fl_epoch, message='_fl_', test_agent1=test_agent1, test_agent2=test_agent2, test_agent3=test_agent3, test_agent4=test_agent4)

    save_all_result(main_agent, assist_agent, oracle_agent, filename='result')



    ###########     PRINT     ##########################

    print('\n ==============        FedSGD      =======================')
    print('\n The federated testing loss', user1.test_loss_fl)
    print('\n The federated testing discount reward', user1.test_discount_rewards_fl)
    print('\n The federated training loss', user1.train_loss_fl)
    print('\n The federated training discount reward', user1.train_discount_rewards_fl)



    print('\n =====================   TEST REWARDS: test 1   ============================')
    print('\n AssistSGD test reward mean at each assist')
    for iii in range(len(main_agent.test_rewards1)):
        print('\n ', np.mean(main_agent.test_rewards1[iii]))
    print('\n FedSGD test reward mean at each round')
    for jjj in range(len(user1.test_rewards1)):
        print('\n ', np.mean(user1.test_rewards1[jjj]))
    print('\n oracle test reward mean', np.mean(oracle_agent.test_rewards1[0]))

    print('\n =====================   TEST REWARDS: test 2   ============================')
    print('\n AssistSGD test reward mean at each assist')
    for iii in range(len(main_agent.test_rewards2)):
        print('\n ', np.mean(main_agent.test_rewards2[iii]))
    print('\n FedSGD test reward mean at each round')
    for jjj in range(len(user1.test_rewards2)):
        print('\n ', np.mean(user1.test_rewards2[jjj]))
    print('\n oracle test reward mean', np.mean(oracle_agent.test_rewards2[0]))

    print('\n =====================   TEST REWARDS: test 3   ============================')
    print('\n AssistSGD test reward mean at each assist')
    for iii in range(len(main_agent.test_rewards3)):
        print('\n ', np.mean(main_agent.test_rewards3[iii]))
    print('\n FedSGD test reward mean at each round')
    for jjj in range(len(user1.test_rewards3)):
        print('\n ', np.mean(user1.test_rewards3[jjj]))
    print('\n oracle test reward mean', np.mean(oracle_agent.test_rewards3[0]))

    print('\n =====================   TEST REWARDS: test 4   ============================')
    print('\n AssistSGD test reward mean at each assist')
    for iii in range(len(main_agent.test_rewards4)):
        print('\n ', np.mean(main_agent.test_rewards4[iii]))
    print('\n FedSGD test reward mean at each round')
    for jjj in range(len(user1.test_rewards4)):
        print('\n ', np.mean(user1.test_rewards4[jjj]))
    print('\n oracle test reward mean', np.mean(oracle_agent.test_rewards4[0]))

    print('\n ====================    Train LOSS        ========================')
    print('\n The train loss at each assist ', main_agent.train_loss_in_assist)
    print('\n Trained loss in FedSGD is', user1.train_loss_fl)
    print('\n The train loss for oracle ', oracle_agent.history_loss_each_epoch[-1])

    print('\n ===================    TRAIN DISCOUNT REWARD   =========================')
    print('\n trained discount reward in AssistSGD is', main_agent.train_discount_reward_in_assistance)
    print('\n trained discount reward in FedSGD is', user1.train_discount_rewards_fl)
    print('\n trained discount reward in oracle', oracle_agent.train_discount_reward_each_epoch[-1])
    end = time.time()
    print('\n The total running time is ' + str((end-start)/60) + ' minutes')

# https://towardsdatascience.com/breaking-down-richard-suttons-policy-gradient-9768602cb63b