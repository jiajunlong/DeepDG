# coding=utf-8
import collections
import os
import sys
import time
import numpy as np
import argparse

from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, img_param_init, print_environ, print_row
from datautil.getdataloader import get_img_dataloader
from extension import normalization
import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam hyper-param')
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int,
                        default=3, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='/home/jiajunlong/Documents/domain/transferlearning/code/DeepDG',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='office')
    parser.add_argument('--data_dir', type=str, default='', help='data dir')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=120, help="max iterations")
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet50',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[0], help='target domains')
    parser.add_argument('--output', type=str,
                        default="train_output", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    normalization.add_arguments(parser)
    args = parser.parse_args()
    normalization.setting(args)

    args.steps_per_epoch = 100
    args.data_dir = args.data_file+args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    args = img_param_init(args)
    args.name = '_'.join([args.domains[args.test_envs[0]], args.norm])
    sys.stdout = Tee(os.path.join(args.output, args.name + '_out.txt'))
    sys.stderr = Tee(os.path.join(args.output, args.name + '_err.txt'))
    print_environ()

    return args


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)

    loss_list = alg_loss_dict(args)
    train_loaders, eval_loaders = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()
    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)

    s = print_args(args, [])
    torch.set_num_threads(8)
    print('=======hyper-parameter used========')
    print(s)

    if 'DIFEX' in args.algorithm:
        ms = time.time()
        n_steps = args.max_epoch*args.steps_per_epoch
        print('start training fft teacher net')
        opt1 = get_optimizer(algorithm.teaNet, args, isteacher=True)
        sch1 = get_scheduler(opt1, args)
        algorithm.teanettrain(train_loaders, n_steps, opt1, sch1)
        print('complet time:%.4f' % (time.time()-ms))

    acc_record = {}
    acc_type_list = ['train', 'valid', 'target']
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc = 0, 0
    print('===========start training===========')
    sss = time.time()

    result = collections.defaultdict(lambda: [])
    last_result_key = None
    for epoch in range(args.max_epoch):
        for iter_num in range(args.steps_per_epoch):
            minibatches_device = [(data)
                                  for data in next(train_minibatches_iterator)]
            if args.algorithm == 'VREx' and algorithm.update_count == args.anneal_iters:
                opt = get_optimizer(algorithm, args)
                sch = get_scheduler(opt, args)
            step_vals = algorithm.update(minibatches_device, opt, sch)

        result['lr'] = opt.param_groups[0]['lr']

        if (epoch in [int(args.max_epoch*0.3), int(args.max_epoch*0.7), int(args.max_epoch*0.9)]) and (not args.schuse):
            for params in opt.param_groups:
                params['lr'] = params['lr']*0.1
            result['lr'] = opt.param_groups[0]['lr']

        if (epoch == (args.max_epoch-1)) or (epoch % args.checkpoint_freq == 0):
            result['epoch'] = epoch
            for item in loss_list:
                result[item] = step_vals[item]
            for item in acc_type_list:
                acc_record[item] = np.mean(np.array([modelopera.accuracy(
                    algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))
                result[item+'_acc'] = acc_record[item]
            if acc_record['valid'] > best_valid_acc:
                best_valid_acc = acc_record['valid']
                target_acc = acc_record['target']
            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_epoch{epoch}.pkl', algorithm, args)
            result['time'] = time.time() - sss
            algorithm_dict = algorithm.state_dict()
            results_keys = sorted(result.keys())
            if results_keys != last_result_key:
                print_row(results_keys, colwidth=12)
                last_result_key = results_keys
            print_row([result[key] for key in results_keys],
                           colwidth=12)

    save_checkpoint('model.pkl', algorithm, args)

    print('valid acc: %.4f' % best_valid_acc)
    print('DG result: %.4f' % target_acc)

    with open(os.path.join(args.output, args.name + '_done.txt'), 'a') as f:
        if args.norm == 'BN':
            f.write('done')
        else:
            f.write(args.norm_cfg)
            f.write('\n')
            f.write(args.replace_norm)
        f.write('\n total cost time:%s\n' % (str(time.time()-sss)))
        f.write('valid acc:%.4f\n' % (best_valid_acc))
        f.write('target acc:%.4f' % (target_acc))
