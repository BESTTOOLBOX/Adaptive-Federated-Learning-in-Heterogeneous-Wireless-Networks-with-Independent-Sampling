from models.text.RNN import RNNModel
from models.text.lstm import ModelLSTMShakespeare
from utils.others import nu_classes
import numpy as np
import torch

def batch_data(data, batch_size, seed):
    """
    data is a dict := {'x': numpy array, 'y': numpy} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data['x'].detach().numpy()
    data_y = data['y'].detach().numpy()
    np.random.seed(seed)
    np.random.shuffle(data_x)
    np.random.seed(seed)
    np.random.shuffle(data_y)

    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

def ravel_model_params(model, grads=False, cuda=False):
    """
    Squash model parameters or gradients into a single tensor.
    """
    if cuda:
        m_parameter = torch.Tensor([0]).cuda()
    else:
        m_parameter = torch.Tensor([0])
    for parameter in list(model.parameters()):
        if grads:
            m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
        else:
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
    return m_parameter[1:]


def unravel_model_params(model, parameter_update):
    """
    Assigns grad_update params to model.parameters.
    This is done by iterating through model.parameters() and assigning the relevant params in grad_update.
    NOTE: this function manipulates model.parameters.
    """
    current_index = 0  # keep track of where to read from grad_update
    for p in model.parameters():
        numel = p.data.numel()
        size = p.data.size()
        p.data.copy_(parameter_update[current_index:current_index + numel].view(size))
        current_index += numel


def init_model(args, n_tokens=None):
    if args.dataset == 'shakespeare':
        net_glob = ModelLSTMShakespeare(args=args).to(args.device)

    elif args.dataset == 'reddit':
        net_glob = RNNModel(rnn_type='LSTM', ntoken=n_tokens,
                            ninp=args.emsize, nhid=args.nhid, nlayers=args.nlayers,
                            dropout=args.dropout, tie_weights=args.tied).to(args.device)

    else:
        exit('Error: Not supported dataset')
    return net_glob

