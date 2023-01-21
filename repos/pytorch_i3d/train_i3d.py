import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, transforms
import videotransforms
from pytorch_i3d import InceptionI3d
from i3d_dataset import I3D_Dataset as Dataset
from i3d_feats_dataset import I3D_Dataset as Feature_Dataset
from Research_Platform import helpers, pytorch_helpers
import mlflow
import os
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser = helpers.bp_parser
parser.add_argument('mode', type=str, help='rgb or flow')
parser.add_argument('root', type=str)
parser.add_argument('config', type=str)
parser.add_argument('nclasses', type=int)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--train-last-layer-only', action='store_true')
# parser.add_argument('--slurm', action='store_true')
parser.add_argument('--window-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.1)

__this_dir__ = os.path.dirname(os.path.abspath(__file__))


def get_datasets(config_file, root, mode, nclasses, batch_size, window_size):
    # print('setting up data loaders')
    if mode != 'features':
        train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                               videotransforms.RandomHorizontalFlip(),
                                               ])
        test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

        dataset = Dataset(config_file, 'train', root, mode, nclasses, train_transforms, window_size=window_size)
        val_dataset = Dataset(config_file, 'validation', root, mode, nclasses, test_transforms, window_size=window_size)
        test_dataset = Dataset(config_file, 'test', root, mode, nclasses, test_transforms, window_size=window_size)
    else:
        dataset = Feature_Dataset(config_file, 'train', root, nclasses, balance_classes=True)
        val_dataset = Feature_Dataset(config_file, 'validation', root, nclasses, balance_classes=False)
        test_dataset = Feature_Dataset(config_file, 'test', root, nclasses, balance_classes=False)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                 pin_memory=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                 pin_memory=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                  pin_memory=False)
    # print('done setting up dataloaders')

    dataloaders = {'train': dataloader, 'val': val_dataloader, 'test': test_dataloader}

    return dataloaders


def run(init_lr=0.1, max_steps=10000, mode='rgb', root='/ssd/Charades_v1_rgb', config_file='charades/charades.json',
        nclasses=158, batch_size=10, train_last_layer_only=False, window_size=64):
    # setup dataset
    dataloaders = get_datasets(config_file, root, mode, nclasses, batch_size, window_size)

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load(os.path.join( __this_dir__, 'models/flow_imagenet.pt')))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load(os.path.join(__this_dir__, 'models/rgb_imagenet.pt')))
    i3d.replace_logits(nclasses)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d = nn.DataParallel(i3d)
    i3d.cuda()

    lr = init_lr
    mlflow.log_param('lr', lr)
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    num_steps_per_update = int(128/(batch_size*len(args.gpu)))
    steps = 0
    # train it
    best_val_f1 = 0
    do_official_test = False
    while steps < max_steps:
        # Each epoch has a training and validation phase
        mlflow.log_metric('step', steps)
        for phase in ['train', 'val', 'test']:
            if phase == 'test' and not do_official_test:
                # print('test, but not doing it because do_official_test is not activated')
                continue
            # print(phase)
            if phase == 'train':
                i3d.train(train_last_layer_only)
            else:
                i3d.eval() # Set model to evaluate mode
                
            tot_loss = helpers.AverageMeter()
            tot_loc_loss = helpers.AverageMeter()
            tot_cls_loss = helpers.AverageMeter()
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            ys = []
            y_s = []
            # print('about to start getting data')
            for data in dataloaders[phase]:
                # print(f'num iter- {num_iter}')
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                if args.train_last_layer_only:
                    t = args.window_size
                else:
                    t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs, args.train_last_layer_only)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss.update(loc_loss.item())

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss.update(cls_loss.item())

                # y_ = torch.argmax(torch.max(per_frame_logits, dim=2)[0], dim=1)
                y_ = np.argmax(np.mean(pytorch_helpers.p2n(per_frame_logits), axis=2), axis=1)
                # y = torch.argmax(torch.max(labels, dim=2)[0], dim=1)
                y = helpers.most_common_in_row(np.argmax(pytorch_helpers.p2n(labels), axis=1))

                ys.extend(y.tolist())
                y_s.extend(y_.tolist())

                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss.update(loss.item())
                if phase == 'train':
                    loss.backward()

                # print(num_steps_per_update, num_iter, phase)
                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    # print(f'backward at steps- {steps}')
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()

            if phase == 'train':
                tot_f1 = f1_score(ys, y_s, average='micro')
                mlflow.log_metric('train loc loss', tot_loc_loss.avg, steps)
                mlflow.log_metric('train cls loss', tot_cls_loss.avg, steps)
                mlflow.log_metric('train total loss', tot_loss.avg, steps)
                mlflow.log_metric('train f1', tot_f1, steps)
                tot_loc_loss.reset()
                tot_cls_loss.reset()
                tot_loss.reset()

            if phase == 'val':
                # print(f'steps (val)- {steps}')
                tot_f1 = f1_score(ys, y_s, average='micro')
                mlflow.log_metric('val loc loss', tot_loc_loss.avg, steps)
                mlflow.log_metric('val cls loss', tot_cls_loss.avg, steps)
                mlflow.log_metric('val total loss', tot_loss.avg, steps)
                mlflow.log_metric('val f1', tot_f1, steps)
                if tot_f1 > best_val_f1:
                    pytorch_helpers.save_torch_statedict(i3d, os.path.join(args.logdir, 'best_model.pth'))
                    mlflow.log_artifact(os.path.join(args.logdir, 'best_model.pth'))
                    best_val_f1 = tot_f1
                    do_official_test = True

            if phase == 'test':
                # print(f'steps (test)- {steps}')
                tot_f1 = f1_score(ys, y_s, average='micro')
                mlflow.log_metric('test loc loss', tot_loc_loss.avg, steps)
                mlflow.log_metric('test cls loss', tot_cls_loss.avg, steps)
                mlflow.log_metric('test total loss', tot_loss.avg, steps)
                mlflow.log_metric('test f1', tot_f1, steps)

            if not (phase == 'test' and not do_official_test):
                fig, ax, cm = helpers.plot_confusion_matrix(np.array(ys), np.array(y_s), classes=np.arange(args.hierarchies_and_nclasses),
                                                            normalize=True)
                fig.savefig(os.path.join(args.logdir, f'confusion_matrix_{phase}.png'))
                mlflow.log_artifact(os.path.join(args.logdir, f'confusion_matrix_{phase}.png'))
                np.save(os.path.join(args.logdir, f'confusion_matrix_{phase}.npy'), cm)
                if phase == 'test':
                    do_official_test = False


if __name__ == '__main__':
    # need to add argparse
    args = parser.parse_args()
    print(args.gpu)
    pytorch_helpers.torch_boiler_plate(args.gpu)
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    # mlflow.set_tracking_uri('/nethome/dscarafoni3/dev/NRI/mlruns')
    mlflow.set_experiment(args.logdir)

    # if args.slurm:
    #     args.logdir = 'logs/' + args.logdir + f'_{os.environ["SLURM_ARRAY_TASK_ID"]}'
    #     helpers.maybe_create_dir(args.logdir)
    # else:
    #     while True:
    #         try:
    #             args.logdir = helpers.next_experiment_dir(args.logdir)
    #             helpers.maybe_create_dir(args.logdir)
    #             break
    #         except Exception as e:
    #             print(f'coudl not get directory {args.logdir}')
    args.logdir = helpers.try_make_next_experiment_dir(args.logdir, nolog=False, slurm=args.slurm)
    print(f'using logdir- {args.logdir}')

    with mlflow.start_run(run_name=':'.join(args.logdir.split('/')[-2:])):
        r = mlflow.active_run()
        helpers.picklesave(os.path.join(args.logdir, 'mlflowr.pkl'), r.info.run_id)
        run(init_lr=args.lr,
            mode=args.mode, root=args.root, config_file=args.config,
            nclasses=args.hierarchies_and_nclasses,
            max_steps=args.epochs,
            train_last_layer_only=args.train_last_layer_only,
            batch_size=args.batch_size,
            window_size=args.window_size)

