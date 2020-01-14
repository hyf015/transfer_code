import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from scipy import ndimage
import math
from tqdm import tqdm
import os
from random import randint
from torchvision.transforms import ToTensor

from floss import floss
from utils import *
from models.late_fusion import late_fusion_meta as late_fusion
from data.lateDataset import lateDataset_meta as lateDataset
from collections import defaultdict

valid_tasks = ['American', 'Pizza', 'Burger', 'Snack', 'Greek', 'Pasta', 'Turkey']
valid_names = ['Alireza', 'Carlos', 'Rahul',]
valid_names_train = ['Carlos', 'Rahul',]
valid_names_test = ['Alireza']
names_for_overall = ['Shaghayegh', 'Yin']

def is_valid(k, valid_names=valid_names):
    for n in valid_names:
        if n in k:
            return True
    return False

def pil_loader_g(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class Tasks(object):
    def __init__(self, feat_path, pred_path, gt_path, valid_names=valid_names):

        self.num_tasks = len(valid_names)
        self.valid_names = valid_names
        self.feat_path = feat_path
        self.pred_path = pred_path
        self.gt_path = gt_path
        feats = os.listdir(feat_path)
        feats = {}
        for n in valid_names:
            feats[n] = sorted([k for k in os.listdir(feat_path) if n in k])

        # Now load in all data into memory for selected tasks
        self.ims = defaultdict(list)
        self.preds = defaultdict(list)
        self.gts = defaultdict(list)
        for n in valid_names:
            for m in feats[n]:
                im = pil_loader_g(os.path.join(feat_path, m))
                im = ToTensor(im)
                self.ims[n].append(im)

                im = pil_loader_g(os.path.join(pred_path, m))
                im = ToTensor(im)
                self.preds[n].append(im)

                imname = m.split('_')
                imname = imname[:2] + 'gt' + imname[2:]
                imname = '_'.join(imname)
                im = pil_loader_g(os.path.join(gt_path, imname))
                self.gts[n].append(im)


        # By default, we just sample disjoint sets from the entire given data
        self.all_indices = {k:list(range(len(v)))
                            for k,v in self.ims.items()}
        self.train_indices = self.all_indices
        self.test_indices = self.all_indices

    def create_sample(self, task, indices, device):
        """Create a sample of a task for meta-learning.
        This consists of a x, y pair.
        """
        feat = [self.ims[task][i] for i in indices]
        pred = [self.preds[task][i] for i in indices]
        gt = [self.gts[task][i] for i in indices]
        
        return {'im': torch.stack(feat, 0).to(device), 'pred': torch.stack(pred,0).to(device), 
        'gt': torch.stack(gt, 0).to(device)}

    def sample(self, num_train=4, num_test=100, device=torch.device('cuda:0')):
        """Yields training and testing samples."""
        picked_task = random.randint(0, self.num_tasks - 1)
        return self.sample_for_task(self.valid_names[picked_task], num_train=num_train,
                                    num_test=num_test, device=device)

    def sample_for_task(self, task, num_train=4, num_test=100, device=device):
        if self.train_indices[task] is self.test_indices[task]:
            # This is for meta-training and meta-validation
            indices = random.sample(self.all_indices[task], num_train + num_test)
            train_indices = indices[:num_train]
            test_indices = indices[-num_test:]
        else:
            # This is for meta-testing
            train_indices = random.sample(self.train_indices[task], num_train)
            test_indices = self.test_indices[task]
        return (self.create_sample(task, train_indices),
                self.create_sample(task, test_indices))


class TestTasks(Tasks):
    """Class for final testing (not testing within meta-learning."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_indices = [indices[:-500] for indices in self.all_indices]
        self.test_indices = [indices[-500:] for indices in self.all_indices]


"""
    Replacement classes for standard PyTorch Module and Linear.
"""


class ModifiableModule(nn.Module):
    def params(self):
        return [p for _, p in self.named_params()]

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self):
        subparams = []
        for name, mod in self.named_submodules():
            for subname, param in mod.named_params():
                subparams.append((name + '.' + subname, param))
        return self.named_leaves() + subparams

    def set_param(self, name, param, copy=False):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in self.named_submodules():
                if module_name == name:
                    mod.set_param(rest, param, copy=copy)
                    break
        else:
            if copy is True:
                setattr(self, name, V(param.data.clone(), requires_grad=True))
            else:
                assert hasattr(self, name)
                setattr(self, name, param)

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            self.set_param(name, param, copy=not same_var)


class GradConv(ModifiableModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        nn.init.normal_(ignore.weight.data, mean=0.0, std=np.sqrt(1. / args[0]))
        nn.init.constant_(ignore.bias.data, val=0)

        self.weights = torch.tensor(ignore.weight.data, requires_grad=True)
        self.bias = torch.tensor(ignore.bias.data, requires_grad=True)

    def forward(self, x):
        return F.conv2d(x, self.weights, self.bias)

    def named_leaves(self):
        return [('weights', self.weights), ('bias', self.bias)]

class GradNorm(ModifiableModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        nn.init.constant_(ignore.weight.data, val=1.0)
        nn.init.constant_(ignore.bias.data, val=0)

        self.weights = torch.tensor(ignore.weight.data, requires_grad=True)
        self.bias = torch.tensor(ignore.bias.data, requires_grad=True)
        self.running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, weight=self.weights, bias=self.bias)

    def named_leaves(self):
        return [('weights', self.weights), ('bias', self.bias), ('running_mean', self.running_mean), 
        'running_var', self.running_var]

"""
    Meta-learnable fully-connected neural network model definition
"""


class GazeEstimationModel(ModifiableModule):
    def __init__(self, ):
        super().__init__()

        # Construct layers
        self.layers = []
        self.layers.append(('fusion.0',GradConv(2, 32, kernel_size=3, padding = 1)))
        self.layers.append(('fusion.1', GradNorm(32)))
        self.layers.append('relu', nn.ReLU())
        self.layers.append(('fusion.3',GradConv(32, 32, kernel_size=3, padding = 1)))
        self.layers.append(('fusion.4', GradNorm(32)))
        self.layers.append('relu', nn.ReLU())

        self.layers.append(('fusion.6',GradConv(32, 8, kernel_size=3, padding = 1)))
        self.layers.append(('fusion.7', GradNorm(8)))
        self.layers.append('relu', nn.ReLU())
        self.layers.append(('fusion.9',GradConv(8, 1, kernel_size=1, padding = 0)))

        # For use with Meta-SGD
        # self.alphas = []
        # if make_alpha:
        #     for i, f_now in enumerate(self.layer_num_features[:-1]):
        #         f_next = self.layer_num_features[i + 1]
        #         alphas = GradLinear(f_now, f_next)
        #         alphas.weights.data.uniform_(0.005, 0.1)
        #         alphas.bias.data.uniform_(0.005, 0.1)
        #         self.alphas.append(('alpha%02d' % (i + 1), alphas))

    def clone(self,):
        new_model = self.__class__(self.activation_type, self.layer_num_features,
                                   make_alpha=make_alpha)
        new_model.copy(self)
        return new_model

    def state_dict(self):
        output = {}
        for key, layer in self.layers:
            output[key + '.weights'] = layer.weights.data
            output[key + '.bias'] = layer.bias.data
        return output

    def load_state_dict(self, weights):
        for key, tensor in weights.items():
            self.set_param(key, tensor, copy=True)

    def forward(self, x):
        for name, layer in self.layers:
            if name == 'relu':
                x = F.relu_(x)
            else:
                x = layer(x)
        x = torch.sigmoid(x)
        return x

    def named_submodules(self):
        return self.layers


"""
    Meta-learning utility functions.
"""


def forward_and_backward(model, data, optim=None, create_graph=False,
                         train_data=None, loss_function=floss().to(torch.device('cuda:0'))):
    model.train()
    if optim is not None:
        optim.zero_grad()
    loss = forward(model, data, train_data=train_data, for_backward=True,
                   loss_function=loss_function)
    loss.backward(create_graph=create_graph, retain_graph=(optim is None))
    if optim is not None:
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
    return loss.data.cpu().numpy()


def forward(model, data, return_predictions=False, train_data=None,
            for_backward=False, loss_function=floss().to(torch.device('cuda:0'))):
    model.train()
    im, pred, gt = data['im'], data['pred'], data['gt']
    y_hat = model(im, pred)
    loss = loss_function(y_hat, gt)
    if return_predictions:
        return y_hat.data.cpu().numpy()
    elif for_backward:
        return loss
    else:
        return loss.data.cpu().numpy()


"""
    Inference through model (with/without gradient calculation)
"""


class MAML(object):
    def __init__(self, model, k, output_dir='save/',
                 train_tasks=None, valid_tasks=None, device=torch.device('cuda:0')):
        self.model = model.to(device)
        self.meta_model = model.clone().to(device)

        self.train_tasks = train_tasks
        self.valid_tasks = valid_tasks
        self.k = k
        self.loss_function = floss().to(device)

        self.output_dir = output_dir

    @property
    def model_parameters_path(self):
        return '%s/meta_learned_parameters.pth.tar' % self.output_dir

    def save_model_parameters(self):
        if self.output_dir is not None:
            torch.save(self.model.state_dict(), self.model_parameters_path)

    def load_model_parameters(self):
        if os.path.isfile(self.model_parameters_path):
            weights = torch.load(self.model_parameters_path)
            self.model.load_state_dict(weights)
            print('> Loaded weights from %s' % self.model_parameters_path)

    def train(self, steps_outer, steps_inner=1, lr_inner=0.01, lr_outer=0.001,
              disable_tqdm=False):
        self.lr_inner = lr_inner
        print('\nBeginning meta-learning for k = %d' % self.k)

        # Outer loop optimizer
        optimizer = torch.optim.Adam(self.model.params(), lr=lr_outer)

        # Model and optimizer for validation
        valid_model = self.model.clone().to(device)
        valid_optim = torch.optim.SGD(valid_model.params(), lr=self.lr_inner)

        for i in tqdm(range(steps_outer), disable=disable_tqdm):
            for j in range(steps_inner):
                # Make copy of main model
                self.meta_model.copy(self.model, same_var=True)

                # Get a task
                train_data, test_data = self.train_tasks.sample(num_train=self.k, device=self.device)

                # Run the rest of the inner loop
                task_loss = self.inner_loop(train_data, self.lr_inner)

            # Calculate gradients on a held-out set
            new_task_loss = forward_and_backward(
                self.meta_model, test_data, train_data=train_data, loss_function=self.loss_function
            )

            # Update the main model
            optimizer.step()
            optimizer.zero_grad()

            if (i + 1) % 100 == 0:
                # Validation
                losses = []
                for j in range(self.valid_tasks.num_tasks):
                    valid_model.copy(self.model)
                    train_data, test_data = self.valid_tasks.sample_for_task(j, num_train=self.k)
                    train_loss = forward_and_backward(valid_model, train_data, valid_optim, loss_function=self.loss_function)
                    valid_loss = forward(valid_model, test_data, train_data=train_data, loss_function=self.loss_function)
                    losses.append((train_loss, valid_loss))
                train_losses, valid_losses = zip(*losses)
                print('%d, meta-valid/train-loss'%i, np.mean(train_losses))
                print('%d, meta-valid/valid-loss'%i, np.mean(valid_losses))

        # Save MAML initial parameters
        self.save_model_parameters()

    def test(self, test_tasks_list, num_iterations=[1, 5, 10], num_repeats=20):
        print('\nBeginning testing for meta-learned model with k = %d\n' % self.k)
        model = self.model.clone().to(self.device)

        # IMPORTANT
        #
        # Sets consistent seed such that as long as --num-test-repeats is the
        # same, experiment results from multiple invocations of this script can
        # yield the same calibration samples.
        # random.seed(4089213955)

        for test_set_name, test_tasks in test_tasks_list.items():
            predictions = OrderedDict()
            losses = OrderedDict([(n, []) for n in num_iterations])
            for i, task_name in enumerate(test_tasks.selected_tasks):
                predictions[task_name] = []
                for t in range(num_repeats):
                    model.copy(self.model)
                    optim = torch.optim.SGD(model.params(), lr=self.lr_inner)

                    train_data, test_data = test_tasks.sample_for_task(i, num_train=self.k)
                    if num_iterations[0] == 0:
                        train_loss = forward(model, train_data, loss_function=self.loss_function)
                        test_loss = forward(model, test_data, train_data=train_data, loss_function=self.loss_function)
                        losses[0].append((train_loss, test_loss))
                    for j in range(np.amax(num_iterations)):
                        train_loss = forward_and_backward(model, train_data, optim, loss_function=self.loss_function)
                        if (j + 1) in num_iterations:
                            test_loss = forward(model, test_data, train_data=train_data, loss_function=self.loss_function)
                            losses[j + 1].append((train_loss, test_loss))

                    # Register ground truth and prediction
                    predictions[task_name].append({
                        'groundtruth': test_data[1].cpu().numpy(),
                        'predictions': forward(model, test_data,
                                               return_predictions=True,
                                               train_data=train_data),
                    })
                    predictions[task_name][-1]['errors'] = angular_error(
                        predictions[task_name][-1]['groundtruth'],
                        predictions[task_name][-1]['predictions'],
                    )

                print('Done for k = %3d, %s/%s... train: %.3f, test: %.3f' % (
                    self.k, test_set_name, task_name,
                    np.mean([both[0] for both in losses[num_iterations[-1]][-num_repeats:]]),
                    np.mean([both[1] for both in losses[num_iterations[-1]][-num_repeats:]]),
                ))

            if self.output_dir is not None:
                # Save predictions to file
                pkl_path = '%s/predictions_%s.pkl' % (self.output_dir, test_set_name)
                with open(pkl_path, 'wb') as f:
                    pickle.dump(predictions, f)

                # Write loss values as plain text too
                np.savetxt('%s/losses_%s_train.txt' % (self.output_dir, test_set_name),
                           [[n, np.mean(list(zip(*v))[0])] for n, v in losses.items()])
                np.savetxt('%s/losses_%s_valid.txt' % (self.output_dir, test_set_name),
                           [[n, np.mean(list(zip(*v))[1])] for n, v in losses.items()])

            out_msg = '> Completed test on %s for k = %d' % (test_set_name, self.k)
            final_n = sorted(num_iterations)[-1]
            final_train_losses, final_test_losses = zip(*(losses[final_n]))
            out_msg += ('\n  at %d steps losses were... train: %.3f, test: %.3f +/- %.3f' %
                        (final_n, np.mean(final_train_losses),
                         np.mean(final_test_losses),
                         np.mean([
                             np.std([
                                 data['errors'] for data in person_data
                             ], axis=0)
                             for person_data in predictions.values()
                         ])))
            print(out_msg)

    def inner_loop(self, train_data, lr_inner=0.01):
        # Forward-pass and calculate gradients on meta model
        loss = forward_and_backward(self.meta_model, train_data,
                                    create_graph=True)

        # Apply gradients
        for name, param in self.meta_model.named_params():
            self.meta_model.set_param(name, param - lr_inner * param.grad)
        return loss



"""
    Actual run script
"""

if __name__ == '__main__':

    # Define and parse configuration for training and evaluations
    parser = argparse.ArgumentParser(description='Meta-learn gaze estimator from RotAE embeddings.')
    
    parser.add_argument('--disable-tqdm', action='store_true',
                        help='Disable progress bar from tqdm (in particular on NGC).')

    parser.add_argument('--load_pretrained_base', action='store_true')


    # Parameters for meta-learning
    parser.add_argument('--gt_path', type=str, default='../gtea_gts')
    parser.add_argument('--pred_path', type=str, default='../gtea2_preds')
    parser.add_argument('--feat_path', type=str, default='../gtea2_feats')
    parser.add_argument('--steps-meta-training', type=int, default=100000,
                        help='Number of steps to meta-learn for (default: 100000)')
    parser.add_argument('--tasks-per-meta-iteration', type=int, default=5,
                        help='Tasks to evaluate per meta-learning iteration (default: 5)')
    parser.add_argument('--lr-inner', type=float, default=1e-5,
                        help='Learning rate for inner loop (for the task) (default: 1e-5)')
    parser.add_argument('--lr-outer', type=float, default=1e-3,
                        help='Learning rate for outer loop (the meta learner) (default: 1e-3)')

    # Evaluation
    parser.add_argument('--skip-training', action='store_true',
                        help='Skips meta-training')
    parser.add_argument('k', type=int,
                        help='Number of calibration samples to use - k as in k-shot learning.')
    parser.add_argument('--num-test-repeats', type=int, default=100,
                        help='Number of times to repeat drawing of k samples for testing '
                             + '(default: 100)')
    parser.add_argument('--steps-testing', type=int, default=1000,
                        help='Number of steps to meta-learn for (default: 1000)')

    args = parser.parse_args()

    # Define data sources (tasks)
    x_keys = 0
    meta_train_tasks = Tasks(args.feat_path, args.pred_path, args.gt_path, valid_names_train)
    meta_val_tasks = Tasks(args.feat_path, args.pred_path, args.gt_path, valid_names_test)
    meta_test_tasks = [
        ('gtea', TestTasks(args.feat_path, args.pred_path, args.gt_path, valid_names_test)),
    ]

    # Construct output directory path string
    output_dir = 'save/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get an example entry to design gaze estimation model
    model = GazeEstimationModel()
    meta_learner = MAML(model, args.k, output_dir,
                                      meta_train_tasks, meta_val_tasks,
                                      device=torch.device('cuda:0'))

    # If doing fine-tuning... try to load pre-trained MLP weights
    if args.load_pretrained_base:
        pretrained_dict = torch.load('save/lf_base.pth.tar')['state_dict']
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('loaded pretrained_dict from save/lf_base.pth.tar')
    # if args.meta_learner == 'NONE' or args.maml_use_pretrained_mlp:
    #     import glob
    #     checkpoint_path = sorted(
    #         glob.glob('%s/checkpoints/at_step_*.pth.tar' % args.input_dir)
    #     )[-1]
    #     weights = torch.load(checkpoint_path)
    #     try:
    #         state_dict = {
    #             'layer01.weights': weights['module.gaze1.weight'],
    #             'layer01.bias': weights['module.gaze1.bias'],
    #             'layer02.weights': weights['module.gaze2.weight'],
    #             'layer02.bias': weights['module.gaze2.bias'],
    #         }
    #         if args.select_z == 'before_z':
    #             state_dict['layer00.weights'] = weights['module.fc_enc.weight']
    #             state_dict['layer00.bias'] = weights['module.fc_enc.bias']
    #     except:  # noqa
    #         state_dict = {
    #             'layer01.weights': weights['gaze1.weight'],
    #             'layer01.bias': weights['gaze1.bias'],
    #             'layer02.weights': weights['gaze2.weight'],
    #             'layer02.bias': weights['gaze2.bias'],
    #         }
    #         if args.select_z == 'before_z':
    #             state_dict['layer00.weights'] = weights['fc_enc.weight']
    #             state_dict['layer00.bias'] = weights['fc_enc.bias']
    #     for key, values in state_dict.items():
    #         model.set_param(key, values, copy=True)
    #     del state_dict
    #     print('Loaded %s' % checkpoint_path)

    if not args.skip_training:
        meta_learner.train(
            steps_outer=args.steps_meta_training,
            steps_inner=args.tasks_per_meta_iteration,
            lr_inner=args.lr_inner,
            lr_outer=args.lr_outer,
            disable_tqdm=args.disable_tqdm,
        )

    # Perform test (which entails the repeated training of person-specific models
    if args.skip_training:
        meta_learner.load_model_parameters()
    meta_learner.lr_inner = args.lr_inner
    meta_learner.test(
        test_tasks_list=OrderedDict(meta_test_tasks),
        num_iterations=list(np.arange(start=0, stop=args.steps_testing + 1, step=20)),
        num_repeats=args.num_test_repeats,
    )
