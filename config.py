import argparse



parser = argparse.ArgumentParser(description='AutoScale for localization based-method')


# Data specifications
parser.add_argument('--train_dataset', type=str, default='ShanghaiA',
                    help='choice train dataset')

parser.add_argument('--task_id', type=str, default='./save_file',
                    help='save checkpoint directory')
parser.add_argument('--workers', type=int, default=4,
                    help='load data workers')
parser.add_argument('--print_freq', type=int, default=30,
                    help='print frequency')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch for training')

# Model specifications
parser.add_argument('--test_dataset', type=str, default='ShanghaiA',
                    help='choice train dataset')
parser.add_argument('--pre', type=str, default='./model/ShanghaiA/model_best.pth',
                    help='pre-trained model directory')


# Optimization specifications
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--area_threshold', type=float, default=0.02,
                    help='area  threshold for training')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4,
                    help='weight decay')
parser.add_argument('--momentum', type=float, default=0.95,
                    help='momentum')
parser.add_argument('--rate_lr', type=float, default=1e-5,
                    help='momentum')
parser.add_argument('--epochs', type=int, default=2000,
                    help='number of epochs to train')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id')


args = parser.parse_args()