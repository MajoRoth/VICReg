import yaml
from pathlib import Path

from types import SimpleNamespace

# Function to convert dictionary to SimpleNamespace recursively
def dict_to_namespace(d):
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_namespace(value)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d



def get_conf(conf_name):
    conf_getter_path = Path(__file__)
    confs = {p.stem: p for p in conf_getter_path.parent.glob("*.yaml")}

    if conf_name not in confs.keys():
      raise Exception(f'you should provide a valid conf name. available confs: {list(confs.keys())}')

    with open(confs[conf_name], 'r') as file:
        cfg = yaml.safe_load(file)
        return dict_to_namespace(cfg)


##############################################################################################
##### Taken from https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder #####
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self
#########################################################################################################
