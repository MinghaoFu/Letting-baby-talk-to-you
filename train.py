import torch 
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.manifold import TSNE
from importlib import import_module

from loss import Loss
from utils import fix_seed, preprocess_args, AverageMeter, accuracy, Log
    
def train_babychillanto(args, model, dataset, log, seed):
    
    criterion = Loss(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(dataset.train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset.val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset.test_dataset, batch_size=args.batch_size, shuffle=False)

    best_model = None
    best_val_performance = float('-inf') 
    for epoch in range(args.n_epochs):
        n_top1 = AverageMeter('Acc@1', ':6.2f')
        train_loss = AverageMeter('Acc@1', ':6.2f')
        n_top2 = AverageMeter('Acc@1', ':6.2f')
        
        cls_train_top1= [AverageMeter('Acc@1', ':6.2f') for _ in range(len(args.labels))]
        cls_train_top2= [AverageMeter('Acc@1', ':6.2f') for _ in range(len(args.labels))]
        cls_val_top1= [AverageMeter('Acc@1', ':6.2f') for _ in range(len(args.labels))]
        cls_val_top2= [AverageMeter('Acc@1', ':6.2f') for _ in range(len(args.labels))]
        cls_test_top1= [AverageMeter('Acc@1', ':6.2f') for _ in range(len(args.labels))]
        cls_test_top2= [AverageMeter('Acc@1', ':6.2f') for _ in range(len(args.labels))]
        
        for batch_data in train_loader:
            b_audios, b_labels, b_ids, b_indices = batch_data
            optimizer.zero_grad()
            outputs = model(b_audios)
            
            _, indices= outputs.topk(k=len(args.labels), dim=1)
            pred = indices[:, 0].squeeze(0)
            log.save_info(outputs, b_labels, b_ids, b_indices, 'train')
                
            loss = criterion(outputs, b_labels)
            loss.backward()
            optimizer.step()
            n_acc1,n_acc2 = accuracy(outputs, b_labels, topk=(1, 2))
            n_top1.update(n_acc1.item(), b_audios.size(0))
            n_top2.update(n_acc2.item(), b_audios.size(0))
            train_loss.update(loss.item(), 1)
            # print(loss.item())
            
            for i, cls in enumerate(args.labels):
                _inds = torch.nonzero(torch.eq(b_labels, i)).view(-1)
                if len(_inds) > 0:
                    cls_acc1,cls_acc2 = accuracy(outputs[_inds], b_labels[_inds], topk=(1, 2))
                    cls_train_top1[i].update(cls_acc1.item(), b_audios.size(0))
                    cls_train_top2[i].update(cls_acc2.item(), b_audios.size(0))
        
        model.eval()
        val_loss = 0
        correct = 0
        id2wrongCount = {id : 0 for id in dataset.val_dataset.data['id'].tolist()}
        cls2wrongCount = {cls : 0 for cls in args.labels}
        
        with torch.no_grad():
            val_top1= AverageMeter('Acc@1', ':6.2f')
            val_top2= AverageMeter('Acc@1', ':6.2f')
            val_loss = AverageMeter('Acc@1', ':6.2f')
            
            for vb_audios, vb_labels, vb_ids, vb_indices in val_loader:
                # Compute predictions and loss
                voutputs = model(vb_audios)
                log.save_info(voutputs, vb_labels, vb_ids, vb_indices, 'val')
                loss = criterion(voutputs, vb_labels)
                
                _, indices= voutputs.topk(k=len(args.labels), dim=1)
                pred = indices[:, 0].squeeze(0)
                #print(indices[torch.nonzero(vb_labels == args.labels.index('deaf'))])
                #print(torch.cat([vb_ids[torch.nonzero(vb_labels == args.labels.index('deaf'))], vb_indices[torch.nonzero(vb_labels == args.labels.index('deaf'))]], dim=1))
                
                correct += pred.eq(vb_labels).sum().item()
    
                correct_indices = torch.nonzero(torch.eq(pred, vb_labels)).view(-1)
                mistake_indices = torch.nonzero(torch.ne(pred, vb_labels)).view(-1)
                if len(mistake_indices) != 0:
                    out_types = [[args.labels[value] for value in row] for row in indices[mistake_indices].tolist()]
                    y_type = [args.labels[value] for value in vb_labels[mistake_indices].tolist()]
                    #print(out_types, y_type)

                    wrong_ids = vb_ids[mistake_indices].tolist()
                    for id in wrong_ids:
                        id2wrongCount[id] += 1
                    wrong_cls = [args.labels[i] for i in vb_labels[mistake_indices].tolist()]
                    for cls in wrong_cls:
                        cls2wrongCount[cls] += 1
                        
                for i, cls in enumerate(args.labels):
                    _inds = torch.nonzero(torch.eq(vb_labels, i)).view(-1)
                    if len(_inds) > 0:
                        cls_acc1,cls_acc2 = accuracy(voutputs[_inds], vb_labels[_inds], topk=(1, 2))
                        cls_val_top1[i].update(cls_acc1.item(), vb_audios.size(0))
                        cls_val_top2[i].update(cls_acc2.item(), vb_audios.size(0))
                    
                val_acc1,val_acc2 = accuracy(voutputs, vb_labels, topk=(1, 2))
                val_top1.update(val_acc1.item(), vb_audios.size(0))
                val_top2.update(val_acc2.item(), vb_audios.size(0))
                val_loss.update(loss.item(), 1)
            
            # for i, cls in enumerate(args.labels):
            #     _inds = torch.nonzero(torch.eq(b_labels, i)).view(-1)
            #     if len(_inds) > 0:
            #         cls_acc1,cls_acc2 = accuracy(outputs[_inds], b_labels[_inds], topk=(1, 2))
            #         cls_train_top1[i].update(cls_acc1.item(), b_audios.size(0))
            #         cls_train_top2[i].update(cls_acc2.item(), b_audios.size(0))
                
        for id, count in id2wrongCount.items():
            id2wrongCount[id] = count / dataset.val_data['id'].eq(id).sum().item() 
            
        for cls, count in cls2wrongCount.items():
            cls_num = dataset.val_data['label'].eq(args.labels.index(cls)).sum().item()
            cls_accuracy = round(count / cls_num, 3)
            cls2wrongCount[cls] = '{} / {} = {}'.format(count, cls_num, cls_accuracy)
        
        id2wrongCount = {k: round(v, 3) for k, v in id2wrongCount.items() if v > 0.3}
        avg_train_acc_top1 = sum([i.avg for i in cls_train_top1]) / len(args.labels)
        avg_train_acc_top2 = sum([i.avg for i in cls_train_top2]) / len(args.labels)
        avg_val_acc_top1 = sum([i.avg for i in cls_val_top1]) / len(args.labels)
        avg_val_acc_top2 = sum([i.avg for i in cls_val_top2]) / len(args.labels)
        if avg_val_acc_top1 > best_val_performance:
            log.save_log(epoch, seed=seed)
            best_val_performance = avg_val_acc_top1
            best_model = model.state_dict() 
            torch.save(best_model, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            
            print(f"Epoch [{epoch+1}/{args.n_epochs}]: Training Loss = {train_loss.avg: .3f}, Train Accuracy Top1/2 = {avg_train_acc_top1: .3f}/{avg_train_acc_top2: .3f}, \
                Val Accuracy Top1/2 = {avg_val_acc_top1: .3f} / {avg_val_acc_top2: .3f}")
            #print('Val: Error rate in each class: {}'.format(cls2wrongCount))
            cls_val_acc_info = 'Validation accuracy top1/2 in each class: ' 
            for i in range(len(args.labels)):
                cls_val_acc_info += '{} : {:.3f} / {:.3f} '.format(args.labels[i], cls_val_top1[i].avg, cls_val_top2[i].avg)
            cls_train_acc_info = 'Training accuracy top1/2 in each class: ' 
            for i in range(len(args.labels)):
                cls_train_acc_info += '{} : {:.3f} / {:.3f} '.format(args.labels[i], cls_train_top1[i].avg, cls_train_top2[i].avg)
            print(cls_val_acc_info)
            print(cls_train_acc_info)
            
        criterion.save_log(train_loss.avg, val_loss.avg, val_top1.avg)
        
        #print(f"Epoch [{epoch+1}/{args.n_epochs}]: Training loss = {train_loss.avg: .3f}, train Accuracy = {n_top1.avg: .3f}/{n_top2.avg: .3f} \
              #Val Accuracy = {val_acc1.item(): .3f} Val Accuracy = {val_top1.avg: .3f}/{val_top2.avg: .3f}, Best Val Accuracy = {best_val_performance: .3f}")
    criterion.plot_loss()
    
    model.load_state_dict(best_model)
    model.eval()
    correct = 0
    inf_times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        test_top1= AverageMeter('Acc@1', ':6.2f')
        test_top2= AverageMeter('Acc@1', ':6.2f')
        test_loss = AverageMeter('Acc@1', ':6.2f')
        for audios, labels, ids, indices in test_loader:
            start.record()
            outputs = model(audios)
            end.record()
            torch.cuda.synchronize()
            inf_times.append(start.elapsed_time(end))

            log.save_info(outputs, labels, ids, indices, 'test')
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            
            test_acc1,test_acc2 = accuracy(outputs, labels, topk=(1, 2))
            test_top1.update(test_acc1.item(), audios.size(0))
            test_top2.update(test_acc2.item(), audios.size(0))
            test_loss.update(loss.item(), 1)
    
    log.save_log(epoch, test=True)
    # test_loss /= len(test_loader.dataset)
    # test_acc = 100 * correct / len(test_loader.dataset)
    
    print(f"Seed = {seed}: Test loss = {test_loss.avg: .3f}, test Accuracy = {test_top1.avg: .3f}/{test_top2.avg: .3f}, average inference time per {args.seg_len}s audio: {sum(inf_times) / len(inf_times): .3f}ms")
    
    return best_model, best_val_performance, test_top1.avg

# def train_donateacry(args, model, dataset):

#     criterion = Loss(args)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)

#     train_loader = DataLoader(dataset.train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
#     val_loader = DataLoader(dataset.val_dataset, batch_size=args.batch_size, shuffle=False)
#     test_loader = DataLoader(dataset.test_dataset, batch_size=args.batch_size, shuffle=False)

#     best_model = None
#     best_val_performance = float('-inf') 
#     for epoch in range(args.n_epochs):
#         n_top1 = AverageMeter('Acc@1', ':6.2f')
#         train_loss = AverageMeter('Acc@1', ':6.2f')
#         n_top2 = AverageMeter('Acc@1', ':6.2f')
#         for b_audios, b_labels, b_ids, b_genders, b_ages in train_loader:
#             optimizer.zero_grad()
#             outputs = model(b_audios)#, b_genders, b_ages)
#             loss = criterion(outputs, b_labels)
#             loss.backward()
#             optimizer.step()
#             n_acc1,n_acc2 = accuracy(outputs, b_labels, topk=(1, 2))
#             n_top1.update(n_acc1.item(), b_audios.size(0))
#             n_top2.update(n_acc2.item(), b_audios.size(0))
#             train_loss.update(loss.item(), 1)
#             # print(loss.item())

#         model.eval()
#         val_loss = 0
#         correct = 0
#         id2wrongCount = {id : 0 for id in dataset.val_dataset.data['id'].tolist()}
#         wrong_ids = []
#         with torch.no_grad():
#             val_top1= AverageMeter('Acc@1', ':6.2f')
#             val_top2= AverageMeter('Acc@1', ':6.2f')
#             val_loss = AverageMeter('Acc@1', ':6.2f')
#             for vb_audios, vb_labels, vb_ids, vb_genders, vb_ages in val_loader:
#                 # Compute predictions and loss
#                 voutputs = model(vb_audios)#, vb_genders, vb_ages)
#                 loss = criterion(voutputs, vb_labels)     
#                 pred = voutputs.argmax(dim=1, keepdim=True)
#                 correct += pred.squeeze(1).eq(vb_labels).sum().item()
#                 correct_indices = torch.nonzero(pred.squeeze(1).eq(vb_labels)).squeeze()
#                 mistake_indices = torch.nonzero(torch.ne(pred.squeeze(1), vb_labels)).squeeze()
#                 #wrong_ids = [vb_ids[mistake_indices]] if len(mistake_indices) == 1 else vb_ids[mistake_indices].tolist()
#                 if len(mistake_indices.shape) != 0:
#                     wrong_ids.extend(vb_ids[mistake_indices].tolist())
                    
#                 for id in wrong_ids:
#                     id2wrongCount[id] += 1
#                 val_acc1,val_acc2 = accuracy(voutputs, vb_labels, topk=(1, 2))
#                 val_top1.update(val_acc1.item(), b_audios.size(0))
#                 val_top2.update(val_acc2.item(), b_audios.size(0))
#                 val_loss.update(loss.item(), 1)


#         # Check if current model has better performance than previous best model
#         if val_top1.avg > best_val_performance:
#             best_val_performance = val_top1.avg
#             best_model = model.state_dict()  # Save the model's state dictionary
#             torch.save(best_model, os.path.join(args.checkpoint_dir, 'best_model.pth'))
        
#         criterion.save_log(train_loss.avg, val_loss.avg, val_top1.avg)
        
#         print(f"Epoch [{epoch+1}/{args.n_epochs}]: Training loss = {train_loss.avg: .3f}, train Accuracy = {n_top1.avg: .3f}/{n_top2.avg: .3f} \
#               Val Accuracy = {val_top1.avg: .3f}/{val_top2.avg: .3f}, Best Val Accuracy = {best_val_performance: .3f}")
#     criterion.plot_loss()
#     return best_model, best_val_performance