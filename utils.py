import numpy as np
import pickle
import random
import os
import torch
import glob

from datetime import datetime

class Log:
    def __init__(self, args):
        self.args = args
        self.err_logs = {'train': [], 'val': [], 'test': []}
        self.nums = {'train': 0, 'val': 0, 'test': 0}
        self.log_file = open(os.path.join(args.checkpoint_dir, 'log.txt'), 'w')
        self.log_file.write('{}\n'.format(str(self.args)))
        
    def save_info(self, outputs, labels, ids, indices, type):
        
        _, inds= outputs.topk(k=len(self.args.labels), dim=1)
        pred = inds[:, 0]
        cor_inds = torch.nonzero(torch.eq(pred, labels)).view(-1)
        err_inds = torch.nonzero(torch.ne(pred, labels)).view(-1)
        
        if len(err_inds) != 0:
            self.nums[type] += len(err_inds)
            err_ids = ids[err_inds]
            err_labels = labels[err_inds]
            self.err_logs[type].append(torch.cat([err_ids.unsqueeze(1), err_labels.unsqueeze(1)], dim=1))
            
            # out_types = [[args.labels[value] for value in row] for row in indices[mistake_indices].tolist()]
            # y_type = [args.labels[value] for value in vb_labels[mistake_indices].tolist()]
            # print(out_types, y_type)

            # wrong_ids = vb_ids[mistake_indices].tolist()
            # for id in wrong_ids:
            #     id2wrongCount[id] += 1
            # wrong_cls = [args.labels[i] for i in vb_labels[mistake_indices].tolist()]
            # for cls in wrong_cls:
            #     cls2wrongCount[cls] += 1
    
    def save_log(self, epoch, seed=None, test=False):
        for k in self.err_logs.keys():
            if self.nums[k] != 0:
                n_err = len(self.err_logs[k])
                n_cor = self.nums[k] - n_err
                acc = 1 - (n_err / self.nums[k])
                err_logs = torch.vstack(self.err_logs[k]).cpu().numpy()
                uniques, counts = np.unique(err_logs, axis=0, return_counts=True)
                self.err_logs[k] = np.concatenate((uniques, np.expand_dims(counts, axis=1)), axis=1)
                acc_info = '{} accuracy: {} / {} = {}'.format(k, n_cor, self.nums[k], acc)
                if test == False:
                    acc_info = 'Epoch [{}/{}] '.format(epoch+1, self.args.n_epochs) + acc_info
                if seed != None:
                    acc_info = 'Seed = {}\n'.format(seed) + acc_info
                self.log_file.write('{}\n'.format(acc_info))
                self.log_file.write('{} errors: {}\n\n\n'.format(k, self.err_logs[k]))
        self.err_logs = {'train': [], 'val': [], 'test': []}
        self.nums = {'train': 0, 'val': 0, 'test': 0}
    
def flatten_list(lst: list):
    return [item for sublist in lst for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]

def center_data(x: np.ndarray):
    return (x - x.mean(axis=0)) / x.std(axis=0)

def preprocess_args(args):
    args.seeds = [int(s) for s in args.seeds.split('+')]
    if args.log_path is None:
        args.log_path = args.checkpoint_path
    args.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.checkpoint_dir = os.path.join('./checkpoint', '{}_{}_{}'.format(args.model, args.seeds[0], args.data))
    if os.path.exists(args.checkpoint_dir) == False:
        os.mkdir(args.checkpoint_dir)
    args.data_dir = os.path.join('./dataset', args.data)

    # labels
    if args.data == 'BabyChillanto':
        args.labels = ['asphyxia', 'deaf', 'hunger', 'normal', 'pain']
    elif args.data == 'DonateACry':
        args.labels = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
        args.n_gender_classes = 2
        args.n_age_classes = 5
    elif args.data == 'Mix':
        args.labels = ['pain', 'hungry', 'asphyxia', 'deaf']
        args.remove_ids = [253]
    else:
        raise ValueError('Unknow dataset {}'.format(args.data))

def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, base_lr, ajust_period=20):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = base_lr * (0.1 ** (epoch // ajust_period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def rename_donateacry(data_dir, tgt_dir=None, baby_id=-1):
    '''
        baby_id: -1 if rename from scratch. If adding new samples to existing dataset, set baby_id as the last id of the existing one.
    '''
    if tgt_dir == None:
        tgt_dir = data_dir
    record2id = {}
    for dir in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, dir)
        if tgt_dir != data_dir:
            os.makedirs(os.path.join(tgt_dir, dir))
        if os.path.isdir(cls_path):
            for file in os.listdir(cls_path):
                if file.lower().endswith(".wav"):
                    if dir.startswith('Full_'):
                        # Baby chillanto
                        record = file[0:2]
                        age = 'Null'
                        gender = 'Null'
                    else:
                        # DonateACry
                        infos = file.split('-')
                        age = infos[-2]
                        gender = infos[-3]
                        record = '-'.join(infos[0:4] + [gender, age])
                        
                    print(record)
                    if record in record2id:
                        record2id[record][1] += 1
                        assigned_id, idx = record2id[record]
                    else:
                        baby_id += 1
                        assigned_id = baby_id
                        idx = 0
                        record2id[record] = [assigned_id, idx]

                    new_file = '-'.join([str(assigned_id), str(idx), gender, age]) + '.wav'
                    src_path = os.path.join(data_dir, dir, file)
                    tgt_path = os.path.join(tgt_dir, dir, new_file)
                    os.rename(src_path, tgt_path)
                    print('Rename {} to {}.'.format(src_path, tgt_path))
                    
if __name__ == '__main__':
    rename_donateacry('/home/minghao.fu/workspace/Letting-baby-talk-to-you/dataset/rename', baby_id=310)

                        