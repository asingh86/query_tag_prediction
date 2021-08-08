import yaml


def read_configs() -> dict:
    """
    This functions read config file
    :return: dict
    """
    with open('./configs/configs.yml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config
