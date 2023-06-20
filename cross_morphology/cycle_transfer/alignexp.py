import os

import gym
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from options import get_options
from cycle.data import CycleData
from cycle.dyncycle import CycleGANModel
from cycle.utils import init_logs
from termcolor import cprint


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)


def add_errors(model, display):
    errors = model.get_current_errors()
    for key, value in errors.items():
        if key == 'G_act_B':
            display += '\n'
        display += '{}:{:.4f}  '.format(key, value)
    return display


def train(args):
    txt_logs, img_logs, weight_logs = init_logs(args)
    data_agent = CycleData(args)  # normalize and initial the pre-collected source and target domain data
    model = CycleGANModel(args)  # initialize all the needed networks
    model.fengine.train_statef(data_agent.data1)  # train the source forward dynamics

    cprint('evaluate the initial transfered policy in the target domain', 'blue')
    model.cross_policy.eval_policy(
        gxmodel=model.netG_B,
        axmodel=model.net_action_G_A,
        eval_episodes=10)

    best_reward = 0

    for iteration in range(5):
        cprint('###### iteration {} #######'.format(iteration + 1), 'red', 'on_blue')

        args.lr_Gx = 1e-4
        args.lr_Ax = 0
        model.update(args)
        end_id = 0
        start_id = end_id
        end_id = start_id + args.pair_n
        cprint('update G phase', 'blue')
        for batch_id in range(start_id, end_id):
            item = data_agent.sample()
            data1, data2 = item
            model.set_input(item)
            model.optimize_parameters()
            real, fake = model.fetch()

            if (batch_id + 1) % args.display_gap == 0:
                display = '\n===> Batch[{}/{}]'.format(batch_id + 1, args.pair_n)
                print(display)
                display = add_errors(model, display)
                txt_logs.write('{}\n'.format(display))
                txt_logs.flush()

                path = os.path.join(img_logs, 'imgA_{}.jpg'.format(batch_id + 1))
                model.visual(path)

            if (batch_id + 1) % args.eval_gap == 0:
                reward = model.cross_policy.eval_policy(
                    gxmodel=model.netG_B,
                    axmodel=model.net_action_G_A,
                    eval_episodes=args.eval_n)
                if reward > best_reward:
                    best_reward = reward
                    model.save(weight_logs)
                print('best_reward:{:.1f}  cur_reward:{:.1f}'.format(best_reward, reward))

        args.init_start = False
        args.lr_Gx = 0
        args.lr_Ax = 1e-4
        model.update(args)
        end_id = 0
        start_id = end_id
        end_id = start_id + args.pair_n
        cprint('update A phase', 'blue')
        for batch_id in range(start_id, end_id):
            item = data_agent.sample()
            data1, data2 = item
            model.set_input(item)
            model.optimize_parameters()
            real, fake = model.fetch()

            if (batch_id + 1) % args.display_gap == 0:
                display = '\n===> Batch[{}/{}]'.format(batch_id + 1, args.pair_n)
                print(display)
                display = add_errors(model, display)
                txt_logs.write('{}\n'.format(display))
                txt_logs.flush()

                path = os.path.join(img_logs, 'imgA_{}.jpg'.format(batch_id + 1))
                model.visual(path)

            if (batch_id + 1) % args.eval_gap == 0:
                reward = model.cross_policy.eval_policy(
                    gxmodel=model.netG_B,
                    axmodel=model.net_action_G_A,
                    eval_episodes=args.eval_n)
                if reward > best_reward:
                    best_reward = reward
                    model.save(weight_logs)

                print('best_reward:{:.1f}  cur_reward:{:.1f}'.format(best_reward, reward))


# def test(args):
#     args.istrain = False
#     args.init_start = False
#     txt_logs, img_logs, weight_logs = init_logs(args)
#     data_agent = CycleData(args)
#     model = CycleGANModel(args)
#     model.fengine.train_statef(data_agent.data1)
#     print(weight_logs)
#     model.load(weight_logs)
#     model.update(args)
#
#     model.cross_policy.eval_policy(
#         gxmodel=model.netG_B,
#         axmodel=model.net_action_G_A,
#         # imgpath=img_logs,
#         eval_episodes=10)


if __name__ == '__main__':
    args = get_options()

    # source env information
    env_name = args.env
    env = gym.make(env_name)
    args.state_dim1 = env.observation_space.shape[0]
    args.action_dim1 = env.action_space.shape[0]
    env.close()

    # target env information
    env_name = args.target_env
    env = gym.make(env_name)
    args.state_dim2 = env.observation_space.shape[0]
    args.action_dim2 = env.action_space.shape[0]
    env.close()

    train(args)
    # args.istrain = False
    # with torch.no_grad():
    #     test(args)
