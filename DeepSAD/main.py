import argparse
import os
import torch
import random
import numpy as np
import json 
import psutil 

from DeepSAD import DeepSAD

def main():

    #-- Set CPU option 
    cpu_set   = 0
    num_cpu   = 96
    p = psutil.Process()
    p.cpu_affinity( [int(num_cpu*cpu_set + i) for i in range(75, num_cpu)] )
    
    
    fix_seed = 2022
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='DeepSAD for time series')

    # basic config
    # parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    # parser.add_argument('--model', type=str, required=True, default='LSTM',
    #                     help='model name, options: [LSTM, TCN]')
    parser.add_argument('--net_name', type=str, required=True, default='LSTM', help='choice of base network')
    parser.add_argument('--load_model', type=str, default=None, help='Model file path (default: None).')
    parser.add_argument('--xp_path', type=str, default='./log/results/', help='Result directory')

    # pretrain 
    parser.add_argument('--pretrain', type=bool, default=True, help='Pretrain neural network parameters via autoencoder.')
    parser.add_argument('--ae_optimizer_name', type=str, default='adam',
                help='Name of the optimizer to use for autoencoder pretraining.')
    parser.add_argument('--ae_lr', type=float, default=0.001,
                help='Initial learning rate for autoencoder pretraining. Default=0.001')
    parser.add_argument('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
    parser.add_argument('--ae_lr_milestone', type=list, default=[5,7],
                help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
    parser.add_argument('--ae_batch_size', type=int, default=32, help='Batch size for mini-batch autoencoder training.')
    parser.add_argument('--ae_weight_decay', type=float, default=1e-6,
                help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/mnt/storage/dataset/htkim/KAMP/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='competition_data.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--timeenc', type=int, default=0, help='type of time feature encoding')

    # classification task 
    parser.add_argument('--seq_len', type=int, default=50, help='input sequence length')

    # model define
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embedding_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--n_features', type=int, default=4, help='number of features')
    parser.add_argument('--n_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--lr_milestone', type=list, default=[5,7], help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
    parser.add_argument('--optimizer_name', type=str, default='adam',
              help='Name of the optimizer to use for Deep SAD network training.')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=5, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    device = torch.device('cuda:{}'.format(args.gpu))
    print(device)
    args.device = device

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    
    # Config setting 
    setting = f'{args.data_path}_{args.net_name}_pre{args.pretrain}_aeep{args.ae_n_epochs}_trep{args.train_epochs}_seq{args.seq_len}_emb{args.embedding_dim}_nf{args.n_features}_nl{args.n_layers}_eta{args.eta}'
    path = os.path.join(args.xp_path, setting)
    if not os.path.exists(path):
        os.makedirs(path)
    
    print('>>>>>>>config setting : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

    # Initialize DeepSAD model and set neural network phi
    deepSAD = DeepSAD(args.eta)
    deepSAD.set_network(args.net_name, args.seq_len, args.n_features, args.embedding_dim, args.batch_size)

    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if args.load_model:
        deepSAD.load_model(model_path=args.load_model, load_ae=True, map_location=args.device)
        print('Loading model from %s.' % args.load_model)

    
    print('Pretraining: %s' % args.pretrain)
    if args.pretrain:
        # Log pretraining details
        print('Pretraining optimizer: %s' % args.ae_optimizer_name)
        print('Pretraining learning rate: %g' % args.ae_lr)
        print('Pretraining epochs: %d' % args.ae_n_epochs)
        print('Pretraining learning rate scheduler milestones: %s' % (args.ae_lr_milestone))
        print('Pretraining batch size: %d' % args.ae_batch_size)
        print('Pretraining weight decay: %g' % args.ae_weight_decay)

        # Pretrain model on dataset (via autoencoder)
        deepSAD.pretrain(
                         optimizer_name=args.ae_optimizer_name,
                         lr=args.ae_lr,
                         n_epochs=args.ae_n_epochs,
                         lr_milestones=args.ae_lr_milestone,
                         batch_size=args.ae_batch_size,
                         weight_decay=args.ae_weight_decay,
                         device=args.device,
                         n_jobs_dataloader=args.num_workers,
                         data=args.data,
                         timeenc=args.timeenc,
                         root_path=args.root_path, 
                         seq_len = args.seq_len, 
                         data_path=args.data_path,
                         n_features=args.n_features,
                         embedding_dim=args.embedding_dim
                         )

        # Save pretraining results
        deepSAD.save_ae_results(path + '/ae_results.json')

    # Log training details
    print('Training optimizer: %s' % args.optimizer_name)
    print('Training learning rate: %g' % args.learning_rate)
    print('Training epochs: %d' % args.train_epochs)
    print('Training learning rate scheduler milestones: %s' % (args.lr_milestone))
    print('Training batch size: %d' % args.batch_size)
    print('Training weight decay: %g' % args.weight_decay)

    # Train model on dataset
    deepSAD.train(  
                optimizer_name=args.optimizer_name,
                lr=args.learning_rate,
                n_epochs=args.train_epochs,
                lr_milestones=args.lr_milestone,
                batch_size=args.batch_size,
                weight_decay=args.weight_decay,
                device=args.device,
                n_jobs_dataloader=args.num_workers,
                data=args.data,
                timeenc=args.timeenc,
                root_path=args.root_path, 
                seq_len = args.seq_len, 
                data_path=args.data_path
                )

    # Test model
    deepSAD.test(device=device, n_jobs_dataloader=args.num_workers)

    # Save results, model, and configuration
    deepSAD.save_results(path + '/results.json')
    deepSAD.save_model(path + '/model.tar')
    with open(f'{path}/config.txt', 'w') as f:
        dic = args.__dict__
        dic['device'] = None  # JSON 출력 불가 에러때문에 device에 None 저장. 
        json.dump(dic, f)



if __name__ == '__main__':
    main()
