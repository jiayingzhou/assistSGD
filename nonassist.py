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

import pickle
import os

parser = argparse.ArgumentParser()
# parser.add_argument('--path', help='path',default=None)
parser.add_argument('--record', help='record', default=False)
parser.add_argument('--device', help='GPU or CPU', default='cuda')
parser.add_argument('--reward_threshold', help='reward', default=200, type=float)
parser.add_argument('--assist_round', help='assistance_round', default=10, type=int)
parser.add_argument('--iteration', help='iteration_round', default=0, type=int)

from gym.envs.box2d.lunar_landerV2 import LunarLander_new

args = parser.parse_args()
record = args.record
reward_threshold = args.reward_threshold
assistance_round = args.assist_round
iteration = args.iteration
device = torch.device(args.device)
# use_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

attach = False

print('record is ', record)
subfile = 'ite'+str(iteration)+'/'
if not os.path.exists('arl_video/'+subfile):
    os.makedirs('arl_video/'+subfile)
if not os.path.exists('result/'+subfile):
    os.makedirs('result/'+subfile)

class Params:
    EPOCH_SIZE_IN_EACH_TRAJECTORY = 20  # How many epochs we want to pack when transferring parameters
    EPOCH_SIZE_IN_SHARE = 10    # How many parameters will be shared during each assistance
    ALPHA = 5e-3       # learning rate
    EPISODE_SIZE_IN_EACH_EPOCH = 64   # how many episodes we want to pack into an epoch
    GAMMA = 0.99        # discount rate
    HIDDEN_SIZE = 64    # number of hidden nodes we have in our dnn
    BETA = 0.1          # the entropy bonus multiplier
    reward_threshold = reward_threshold   # The stopping rule: when the last mean reward from 100 episodes > reward_threshold, stop!
    record = record   # whether to record the video or not
    device = device

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
    def __init__(self, environment: str = "CartPole",  height_level=0.25,LEG_SPRING_TORQUE=40, map_prob=1, map_n=10, msg1=None, msg2=None):

        self.ALPHA = Params.ALPHA
        self.EPISODE_SIZE_IN_EACH_EPOCH = Params.EPISODE_SIZE_IN_EACH_EPOCH
        self.GAMMA = Params.GAMMA
        self.HIDDEN_SIZE = Params.HIDDEN_SIZE
        self.BETA = Params.BETA
        self.EPOCH_SIZE_IN_EACH_TRAJECTORY = Params.EPOCH_SIZE_IN_EACH_TRAJECTORY
        self.EPOCH_SIZE_IN_SHARE = Params.EPOCH_SIZE_IN_SHARE
        self.reward_threshold = Params.reward_threshold
        # self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.DEVICE = Params.device
        self.BATCH_num = 0
        self.exit_flag = False
        self.msg1 = msg1
        self.msg2 = msg2
        self.record = record
        # instantiate the tensorboard writer
        # self.writer = SummaryWriter(comment=f'_PG_CP_Gamma={self.GAMMA},'
        #                                     f'LR={self.ALPHA},'
        #                                     f'BS={self.BATCH_SIZE},'
        #                                     f'NH={self.HIDDEN_SIZE},'
        #                                     f'BETA={self.BETA}')

        # create the environment
        self.env = environment
        self.LEG_SPRING_TORQUE = LEG_SPRING_TORQUE
        self.height_level = height_level
        self.map_prob = map_prob
        self.map_n = map_n
        self.episode_threshold = 0
        self.epoch_threshold = 0


        # the agent driven by a neural network architecture
        self.agent = Agent(observation_space_size=self.env.observation_space.shape[0],
                           action_space_size=self.env.action_space.n,
                           hidden_size=self.HIDDEN_SIZE).to(self.DEVICE)

        self.opt = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)
        # the total_rewards record mean rewards in the latest 100 episodes
        self.total_rewards_each_episode = deque([], maxlen=100)
        if attach:
            self.rewards_each_epoch = []
            self.history_loss = deque([], maxlen=100000)
            self.test_rewards = deque([], maxlen=100000)
            self.test_loss = deque([], maxlen=100000)

            self.total_model_in_share = deque([], maxlen=self.EPOCH_SIZE_IN_SHARE)
        # This is for selecting parameters, different parameters in different models
            self.total_model_in_share.append(self.agent)

        # flag to figure out if we have render a single episode current epoch
        self.finished_rendering_this_epoch = False


    def test_on_environment(self, provider, test=True):
        epoch = 0
        test_loss = []
        if self.record:
            provider.env = wrappers.Monitor(provider.env, './arl_video/' + subfile+ self.msg1 + '/' + self.msg2+'_on_' + provider.msg2, video_callable=lambda episode_id: True,
                                   force=True)
        while epoch < self.epoch_threshold:
            loss, _ = provider.play_epoch_use_agent(self, test)
            # if the epoch is over - we have epoch trajectories to perform the policy gradient

            # reset the rendering flag
            self.finished_rendering_this_epoch = False
            epoch += 1

            test_loss.append(loss.data)
        if self.record:
            self.env = self.env.unwrapped
        self.env.close()
        return test_loss


    def get_assistance_from(self, provider, assist_round):
        """
                    The main interface for the Policy Gradient solver
                """


        # init the epoch arrays
        # used for entropy calculation
        self.memory_loss = deque([], maxlen=self.EPOCH_SIZE_IN_SHARE)
        self.total_model_in_share = deque([], maxlen=self.EPOCH_SIZE_IN_SHARE)

        # choose parameter who yield the smallest loss
        # We use all candidate parameters to perform episodes, and calculate loss (loss1 + loss2)

        # self.BATCH_num is the nmber of batches (number of assitance round. when it equals 0, no need )
        if self.BATCH_num > 0:
            for param_cand in range(len(provider.total_model_in_share)):
                provider.agent = provider.total_model_in_share[param_cand]
                min_loss = np.Inf
                for episode_idx in range(self.EPOCH_SIZE_IN_SHARE):
                    loss, _ = self.play_epoch_use_agent(provider)
                    loss = loss + provider.memory_loss[param_cand]
                    if loss < min_loss:
                        max_ind = param_cand
                        min_loss = loss
                # decide the parameters and use it as starting point
                provider.agent = provider.total_model_in_share[max_ind]





        # init the episode and the epoch
        # init local training
        epoch = 0
        loss_in_accumulate_epoch = deque([], maxlen=100)

        # launch a monitor at each round of assistance
        if self.record:
            self.env = wrappers.Monitor(self.env, './arl_video/' + subfile + self.msg1 + '/'+ self.msg2 + '/'+ str(assist_round), video_callable=lambda episode_id: True,
                                   force=True)

        while epoch < self.EPOCH_SIZE_IN_EACH_TRAJECTORY:
            if epoch == 0:
                loss, reward_epoch_mean = self.play_epoch_use_agent(provider)

            else:
                loss, reward_epoch_mean = self.play_epoch_use_agent(self)

            loss_in_accumulate_epoch.append(loss.data.cpu().numpy())
            # append loss and parameters for next round of assitance: user assist provider
            if attach:
                self.history_loss.append(loss.data)
            if epoch % self.EPOCH_SIZE_IN_SHARE == 0:
                self.memory_loss.append(loss)
                self.total_model_in_share.append(self.agent)
            # increment the epoch
            epoch += 1
            print('epoch ', epoch)
            #  convergence rule
            # if epoch > 10:
            #     if np.std(loss_in_accumulate_epoch)/np.sqrt(len(loss_in_accumulate_epoch)) < self.epoch_threshold:
            #         break
            if np.mean(self.total_rewards_each_episode) > self.reward_threshold:
                break

            # reset the rendering flag
            self.finished_rendering_this_epoch = False


            # zero the gradient
            self.opt.zero_grad()
            # backprop
            loss.backward()
            # update the parameters
            self.opt.step()

            # feedback
            print("\r", f"Epoch: {epoch + self.BATCH_num * self.EPOCH_SIZE_IN_EACH_TRAJECTORY}, Avg Return per Epoch: {reward_epoch_mean:.3f}",
                  end="",
                  flush=True)
            del loss


        # unwrap the monitor for next round of wrapping
        if self.record:
            self.env = self.env.unwrapped
        # self.env.close()

        self.BATCH_num += 1
    # close the environment
    #     self.env.close()

        # close the writer
        # self.writer.close()

    def self_train(self, tic):
        """
                    The main interface for the Policy Gradient solver
                """


        # init the epoch arrays
        # used for entropy calculation
        if attach:
            self.memory_loss = deque([], maxlen=self.EPOCH_SIZE_IN_SHARE)
            self.total_model_in_share = deque([], maxlen=self.EPOCH_SIZE_IN_SHARE)

        # choose parameter who yield the smallest loss
        # We use all candidate parameters to perform episodes, and calculate loss (loss1 + loss2)

        # self.BATCH_num is the nmber of batches (number of assitance round. when it equals 0, no need )

        # init the episode and the epoch

        epoch = 0
        if attach:
            loss_in_accumulate_epoch = deque([], maxlen=100)
        if self.record:
            self.env = wrappers.Monitor(self.env, './arl_video/' + subfile + self.msg1 + '/'+ self.msg2, video_callable=lambda episode_id: True,
                                   force=True)
        while epoch < self.EPOCH_SIZE_IN_EACH_TRAJECTORY:
            loss, reward_epoch_mean = self.play_epoch_use_agent(self)
            if attach:
                loss_in_accumulate_epoch.append(loss.data.cpu().numpy())

            # if the epoch is over - we have epoch trajectories to perform the policy gradient


            # reset the rendering flag
            self.finished_rendering_this_epoch = False
            # append loss and parameters for next round of assitance: user assist provider
            if attach:
                self.history_loss.append(loss.data)

            if epoch % self.EPOCH_SIZE_IN_SHARE == 0:
                if attach:
                    self.memory_loss.append(loss)
                    self.total_model_in_share.append(self.agent)
                toc = time.time()
                print('The running minute is ', (toc-tic)/60)
                # t = torch.cuda.get_device_properties(0).total_memory
                f = torch.cuda.max_memory_allocated(0) - torch.cuda.memory_allocated(0)  # free inside reserved
                print('memory left is ', f)
                torch.cuda.empty_cache()
                f = torch.cuda.max_memory_allocated(0) - torch.cuda.memory_allocated(0) # free inside reserved
                print('memory left is ', f)


            # increment the epoch
            epoch += 1
            # zero the gradient
            self.opt.zero_grad()
            # backprop
            loss.backward()
            # update the parameters
            self.opt.step()
            # feedback
            print("\r", f"Epoch: {epoch + self.BATCH_num * self.EPOCH_SIZE_IN_EACH_TRAJECTORY}, Avg Return per Epoch: {reward_epoch_mean:.3f}",
                  end="",
                  flush=True)
            del loss, reward_epoch_mean


            # self.writer.add_scalar(tag='Average Return over 100 episodes',
            #                        scalar_value=np.mean(self.total_rewards),
            #                        global_step=epoch)
            # check if solved
            # if epoch > 10:
            #     if np.std(loss_in_accumulate_epoch)/np.sqrt(len(loss_in_accumulate_epoch)) < self.epoch_threshold:
            #         break
            if np.mean(self.total_rewards_each_episode) > self.reward_threshold:
                break

        if self.record:
            self.env = self.env.unwrapped

        self.BATCH_num += 1
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

        which_map = Categorical(self.map_prob).sample()

        height = self.height_level[which_map]

        index_inside_map = self.map_n[which_map]

        seed = index_inside_map[torch.randperm(len(index_inside_map))[0]]

        self.env.seed(int(seed))

        state = self.env.reset(height_level=height, LEG_SPRING_TORQUE=self.LEG_SPRING_TORQUE)

        # initialize the episode arrays
        episode_actions = torch.empty(size=(0,), dtype=torch.long, device=self.DEVICE)
        episode_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        cumulative_average_rewards = np.empty(shape=(0,), dtype=float)
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
                if standardize:
                    discounted_rewards_to_go -= cumulative_average_rewards  # baseline - state specific average

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
                sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

                # won't render again this epoch
                self.finished_rendering_this_epoch = True
                # env_wrap.close()
                self.env.close()

                del episode_actions, episode_weighted_log_probs, mask, episode_log_probs,discounted_rewards_to_go,cumulative_average_rewards
                del which_map, height, index_inside_map, seed, state, action, reward, done
                return sum_weighted_log_probs, episode_logits, sum_of_rewards

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
        entropy_bonus = -1 * self.BETA * entropy
        del epoch_logits, p, log_p
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

    def play_epoch_use_agent(self, provider, test=False):

        epoch_logits = torch.empty(size=(0, self.env.action_space.n), device=self.DEVICE)
        epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)
        loss_in_accumulate_episode = np.empty(shape=(0,), dtype=float)
        mean_loss_in_accumulate_episode = np.empty(shape=(0,), dtype=float)
        episode = 0
        rewards_epoch = np.empty(shape=(0,), dtype=float)
        while episode < self.EPISODE_SIZE_IN_EACH_EPOCH:
            (episode_weighted_log_prob_trajectory,
             episode_logits,
             sum_of_episode_rewards
             ) = self.play_episode_use_agent(provider)

            episode += 1

            # after each episode append the sum of total rewards to the deque
            if attach:
                if test:
                    self.test_rewards.append(sum_of_episode_rewards)
                else:
                    self.total_rewards_each_episode.append(sum_of_episode_rewards)


            # append the weighted log-probabilities of actions
            epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),
                                                 dim=0)

            # append the logits - needed for the entropy bonus calculation
            epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)
            rewards_epoch = np.concatenate((rewards_epoch, np.array([sum_of_episode_rewards])), axis=0)

            # calculate the loss
            loss, entropy = self.calculate_loss(epoch_logits=epoch_logits,
                                                weighted_log_probs=epoch_weighted_log_probs)

            loss_in_accumulate_episode = np.concatenate((loss_in_accumulate_episode, [loss.cpu().data.numpy()]), axis=0)

            mean_loss_in_accumulate_episode = np.concatenate((mean_loss_in_accumulate_episode, [np.mean(loss_in_accumulate_episode)]), axis=0)
            # print("end in episode",
            #       np.std(mean_loss_in_accumulate_episode) / np.sqrt(mean_loss_in_accumulate_episode.shape[0]))
            # check if converging
            if episode >= 10:
                if np.std(mean_loss_in_accumulate_episode)/np.sqrt(mean_loss_in_accumulate_episode.shape[0]) < self.episode_threshold:
                    print('episode stop, and is', episode)
                    break

        # reset the rendering flag
        if not test:
            if attach:
                self.rewards_each_epoch.append(np.mean(rewards_epoch))
        self.finished_rendering_this_epoch = False
        loss, entropy = self.calculate_loss(epoch_logits=epoch_logits,
                                            weighted_log_probs=epoch_weighted_log_probs)
        mean_rewards_epoch = np.mean(rewards_epoch)
        del loss_in_accumulate_episode, mean_loss_in_accumulate_episode, epoch_weighted_log_probs, episode_logits, entropy,rewards_epoch
        # if the epoch is over - we have epoch trajectories to perform the policy gradient
        return loss, mean_rewards_epoch


# def convergence(loss_vec):

def train_assist(user_map, provider_map, test_map, message=None):
    # env = gym.make('LunarLander-v2')
    env = LunarLander_new()
    # env = wrappers.Monitor(env, './arl_video/' + str(uuid.uuid4()), video_callable=lambda episode_id: True, force=True)
    # use_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    map1 = user_map.steep_level
    map1_prob = user_map.prob
    map1_n = user_map.quantity
    map2 = provider_map.steep_level
    map2_prob = provider_map.prob
    map2_n = provider_map.quantity
    map_test = test_map.steep_level
    maptest_prob = test_map.prob
    maptest_n = test_map.quantity
    # args = parser.parse_args()
    # env = args.env
    # use_cuda = args.use_cuda

    # assert(env in ['CartPole', 'LunarLander'])

    main_agent = PolicyGradient(environment=env, height_level=map1, LEG_SPRING_TORQUE=40, map_prob=map1_prob, map_n=map1_n, msg1=message, msg2='u')
    assist_agent = PolicyGradient(environment=env, height_level=map2, LEG_SPRING_TORQUE=40, map_prob= map2_prob, map_n=map2_n, msg1=message,msg2='p')
    test_agent = PolicyGradient(environment=env,  height_level=map_test, LEG_SPRING_TORQUE=40,map_prob=maptest_prob, map_n=maptest_n, msg1=message, msg2='t')
    communication_round = assistance_round
    for communication in range(communication_round):
        main_agent.get_assistance_from(assist_agent, assist_round=communication)
        f = open('result/'+subfile+'assist', 'wb')
        main_param_list = []
        for name, param in main_agent.agent.named_parameters():
            main_param_list.append(param)
        assist_param_list = []
        for name, param in assist_agent.agent.named_parameters():
            assist_param_list.append(param)
        pickle.dump(main_agent.total_rewards_each_episode, f)
        pickle.dump(assist_agent.total_rewards_each_episode, f)
        if attach:
            pickle.dump(main_agent.rewards_each_epoch, f)
            pickle.dump(assist_agent.rewards_each_epoch, f)
        pickle.dump(main_param_list, f)
        pickle.dump(assist_param_list, f)
        if attach:
            pickle.dump(main_agent.history_loss, f)
            pickle.dump(assist_agent.history_loss, f)
            pickle.dump(main_agent.test_rewards, f)
            pickle.dump(assist_agent.test_rewards, f)
            pickle.dump(main_agent.test_loss, f)
            pickle.dump(assist_agent.test_loss, f)
        f.close()
        assist_agent.get_assistance_from(main_agent, assist_round=communication)


    main_agent.main_on_test_loss = main_agent.test_on_environment(test_agent)

    return main_agent, assist_agent


def train_single(user_map, test_map, message=None):
    # env = gym.make('LunarLander-v2')
    env = LunarLander_new()
    # use_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('use suda ', use_cuda)

    map1 = user_map.steep_level
    map1_prob = user_map.prob
    map1_n = user_map.quantity

    map_test = test_map.steep_level
    maptest_prob = test_map.prob
    maptest_n = test_map.quantity

    main_agent = PolicyGradient(environment=env,  height_level=map1, LEG_SPRING_TORQUE=40, map_prob=map1_prob, map_n=map1_n, msg1=message, msg2='u')
    test_agent = PolicyGradient(environment=env,  height_level=map_test, LEG_SPRING_TORQUE=40, map_prob=maptest_prob, map_n=maptest_n, msg1=message, msg2='t')
    tic = time.time()
    for communication in np.arange(assistance_round):
        main_agent.self_train(tic=tic)
        communication += 1

    f = open('result/'+subfile+'non_assist', 'wb')
    main_param_list = []
    for name, param in main_agent.agent.named_parameters():
        main_param_list.append(param)
    pickle.dump(main_param_list, f)
    if attach:
        pickle.dump(main_agent.history_loss, f)
        pickle.dump(main_agent.test_rewards, f)
        pickle.dump(main_agent.test_loss, f)
    f.close()



    main_agent.main_on_test_loss = main_agent.test_on_environment(test_agent)
    del main_param_list, name, param, f
    # main_agent.main_on_main_rewards, main_agent.main_on_main_loss = main_agent.test_on_environment(main_agent)

    return main_agent

if __name__ == "__main__":
    times = 10
    n1 = [10, 2]
    n2 = [50, 50]
    nt = [10*times, 10*times]
    #
    # class map_user:
    #     steep_level = torch.tensor([0.2, 0.6])
    #     prob = torch.tensor(n1)/sum(n1)
    #     quantity = [np.arange(0, n1[0]), np.arange(0, n1[1])]
    # class map_provider:
    #     steep_level = torch.tensor([0.2, 0.6])
    #     prob = torch.tensor(n2)/sum(n2)
    #     quantity = [np.arange(n1[0], n1[0]+n2[0]), np.arange(n1[1], n1[1]+n2[1])]
    # class map_test:
    #     steep_level = torch.tensor([0.2, 0.6])
    #     prob = torch.tensor(nt)/sum(nt)
    #     quantity = [np.arange(n1[0]+n2[0], n1[0]+n2[0]+nt[0]), np.arange(n1[1]+n2[1], n1[1]+n2[1]+nt[1])]
    class map_user:
        steep_level = torch.tensor([0.2, 0.6])
        prob = torch.tensor([n1[0]/(n1[0]+n1[1]), n1[1]/(n1[0]+n1[1])])
        quantity = [np.arange(0, n1[0]), np.arange(0, n1[1])]
    class map_provider:
        steep_level = torch.tensor([0.2, 0.6])
        prob = torch.tensor([n2[0]/(n2[0]+n2[1]), n2[1]/(n2[0]+n2[1])])
        quantity = [np.arange(n1[0], n1[0]+n2[0]), np.arange(n1[1], n1[1]+n2[1])]
    class map_test:
        steep_level = torch.tensor([0.2, 0.6])
        prob = torch.tensor([nt[0]/(nt[0]+nt[1]), nt[1]/(nt[0]+nt[1])])
        quantity = [np.arange(n1[0]+n2[0], n1[0]+n2[0]+nt[0]), np.arange(n1[1]+n2[1], n1[1]+n2[1]+nt[1])]
    start = time.time()
    # main_agent, assist_agent = train_assist(map_user, map_provider, map_test, message='_assist_')


    single_agent = train_single(map_user, map_test, message='_nonasssit_')
    end = time.time()
    print('The total running time is ' + str((end-start)/60) + ' minutes')
    f = open('result/'+subfile+'result', 'wb')
    main_param_list = []
    # for name, param in main_agent.agent.named_parameters():
    #     main_param_list.append(param)
    # assist_param_list = []
    # for name, param in assist_agent.agent.named_parameters():
    #     assist_param_list.append(param)
    # single_param_list = []
    # for name, param in single_agent.agent.named_parameters():
    #     single_param_list.append(param)
    # pickle.dump(main_agent.total_rewards_each_episode, f)
    # pickle.dump(assist_agent.total_rewards_each_episode, f)
    # pickle.dump(main_agent.rewards_each_epoch, f)
    # pickle.dump(assist_agent.rewards_each_epoch, f)
    pickle.dump(main_param_list, f)
    # pickle.dump(assist_param_list, f)
    # pickle.dump(single_param_list, f)
    # pickle.dump(main_agent.history_loss, f)
    # pickle.dump(assist_agent.history_loss, f)
    if attach:
        pickle.dump(single_agent.history_loss, f)
    # pickle.dump(main_agent.test_rewards, f)
    # pickle.dump(assist_agent.test_rewards, f)
        pickle.dump(single_agent.test_rewards, f)
    # pickle.dump(main_agent.test_loss, f)
    # pickle.dump(assist_agent.test_loss, f)
        pickle.dump(single_agent.test_loss, f)
    f.close()

# https://towardsdatascience.com/breaking-down-richard-suttons-policy-gradient-9768602cb63b