import time

import click

from reranking_models import *
from util_crosscbr import *
from model_crosscbr import *

import yaml
import glob_settings

CUDA = torch.cuda.is_available()
TRN_DEVICE = torch.device('cuda' if CUDA else 'cpu')
EVA_DEVICE = torch.device('cuda')

def set_seed(seed):
    """
    Set random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_mat_dataset(dataname):
    """
    Load dataset
    """
    path = f'../data_pkl/{dataname}'
    # conf = yaml.safe_load(open("./config.yaml"))
    conf = glob_settings.config_global
    # conf = conf[dataname]
    # conf['dataset'] = dataname
    # conf['device'] = TRN_DEVICE
    
    dataset = Datasets(conf)
    
    n_user, n_bundle, n_item = dataset.num_users, dataset.num_bundles, dataset.num_items
    _, user_bundle_trn = dataset.get_ub('train')
    _, user_bundle_vld = dataset.get_ub('tune')
    _, user_bundle_test = dataset.get_ub('test')
    _, user_item = dataset.get_ui()
    bundle_item = dataset.get_bi()
    
    user_bundle_neg = dataset.user_bundle_neg
    
    
    user_bundle_test_mask = user_bundle_trn + user_bundle_vld

    # filtering
    user_bundle_vld, vld_user_idx = user_filtering(user_bundle_vld,
                                                   user_bundle_neg)

    model = CrossCBR(conf, dataset.graphs).to(TRN_DEVICE)
    
    return n_user, n_item, n_bundle, bundle_item, user_item,\
           user_bundle_trn, user_bundle_vld, vld_user_idx, user_bundle_test,\
           user_bundle_test_mask, model


def user_filtering(csr, neg):
    """
    Aggregate ground-truth targets and negative targets
    """
    idx, _ = np.nonzero(np.sum(csr, 1))
    pos = np.nonzero(csr[idx].toarray())[1]
    pos = pos[:, np.newaxis]
    neg = neg[idx]
    arr = np.concatenate((pos, neg), axis=1)
    return arr, idx


@click.command()
@click.option('--data', type=str, default='Youshu')
@click.option('--base', type=str, default='CrossCBR')
@click.option('--model', type=str, default='popcon')
@click.option('--beta', type=float, default=10000)
@click.option('--n', type=int, default=200)
@click.option('--seed', type=int, default=0)
def main(data, base, model, beta, n, seed):
    """
    Main function
    """
    set_seed(seed)
    n_user, n_item, n_bundle, bundle_item, user_item,\
    user_bundle_trn, user_bundle_vld, vld_user_idx, user_bundle_test,\
    user_bundle_test_mask, model = load_mat_dataset(data)
    ks = [30, 50]
    result_path = f'./checkpoints/{data}/{base}/model/results.pt'
    results = model.load_state_dict(torch.load(result_path).to('cpu'))
    
    print('=========================== LOADED ===========================')

    if model == 'origin':
        model = Origin()
        model.get_dataset(n_user, n_item, n_bundle, bundle_item, user_item,
                          user_bundle_trn, user_bundle_vld, vld_user_idx,
                          user_bundle_test, user_bundle_test_mask)

    elif model == 'popcon':
        model = PopCon(beta=beta, n=n)
        model.get_dataset(n_user, n_item, n_bundle, bundle_item, user_item,
                          user_bundle_trn, user_bundle_vld, vld_user_idx,
                          user_bundle_test, user_bundle_test_mask)

    test_start_time = time.time()
    test_recalls, test_maps, test_covs, test_ents, test_ginis = model.evaluate_test(results, ks, div=True)
    test_elapsed = time.time() - test_start_time

    test_content = form_content(0, [0, 0],
                                test_recalls, test_maps, test_covs, test_ents, test_ginis,
                                [0, test_elapsed, test_elapsed])
    print(test_content)


def form_content(epoch, losses, recalls, maps, covs, ents, ginis, elapses):
    """
    Format of logs
    """
    content = f'{epoch:7d}| {losses[0]:10.4f} {losses[1]:10.4f} |'
    for item in recalls:
        content += f' {item:.4f} '
    content += '|'
    for item in maps:
        content += f' {item:.4f} '
    content += '|'
    for item in covs:
        content += f' {item:.4f} '
    content += '|'
    for item in ents:
        content += f' {item:.4f} '
    content += '|'
    for item in ginis:
        content += f' {item:.4f} '
    content += f'| {elapses[0]:7.1f} {elapses[1]:7.1f} {elapses[2]:7.1f} |'
    return content


if __name__ == '__main__':
    main()
