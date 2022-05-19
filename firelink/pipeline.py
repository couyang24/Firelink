import pickle
import yaml

from sklearn.pipeline import Pipeline


class FirePipeline(Pipeline):
    """wrapper of sklearn pipeline"""

    def __init__(self, steps):
        super().__init__(steps)

    @classmethod
    def link_fire(cls, file):
        """load pickled file"""
        with open(file, "rb") as obj:
            return pickle.load(obj)

    @classmethod
    def link_yaml(cls, file):
        """load yaml file"""
        with open(file,'rb') as yml:
            cur_yaml = yaml.safe_load(yml)
        pipe_lst = []
        for key in cur_yaml.keys():
            method = eval(key.split('_')[0])
            config = cur_yaml[key]
            pipe_lst += [
                (
                    key,
                    method(config['condition'], config['target'], config['value'])
                )
            ]
        return FirePipeline(pipe_lst)

    def save_fire(self, file, file_type="ember"):
        if file_type == "ember":
            with open(file, "wb") as obj:
                pickle.dump(self, obj)
