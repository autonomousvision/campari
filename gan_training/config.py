import yaml
from gan_training import campari
from gan_training.metrics import Evaluator

method_dict = {
    'campari': campari,
}


def get_dict_from_yaml_file(yaml_file):

    with open(yaml_file) as f:
        d = yaml.safe_load(f)
    return d


def process_config(config_file, default_config_file='configs/default.yaml'):
    d_base = get_dict_from_yaml_file(default_config_file)
    d_cur = get_dict_from_yaml_file(config_file)

    # Check if we should inherit from a config
    inherit_from = d_cur.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        d_inherit = get_dict_from_yaml_file(inherit_from)
        update_recursive(d_base, d_inherit)

    update_recursive(d_base, d_cur)
    return d_base

    
def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def get_model(cfg, mode="train"):
    method_name = cfg['method']
    model = method_dict[method_name].config.get_model(cfg, mode=mode)
    return model


def get_trainer(model, evaluator, optimizer_g, optimizer_d, cfg):
    method_name = cfg['method']
    model = method_dict[method_name].config.get_trainer(
        model, evaluator, optimizer_g, optimizer_d, cfg)
    return model


def get_optimizer(model, cfg):
    method_name = cfg['method']
    optimizer_g, optimizer_d = method_dict[method_name].config.get_optimizer(
        model, cfg)
    return optimizer_g, optimizer_d


def get_evaluator(cfg, dataloader=None):
    dataset_name = cfg['data']['dataset']
    img_size = cfg['data']['img_size']
    evaluator = Evaluator(dataset_name=dataset_name, img_size=img_size, dataloader=dataloader)
    return evaluator