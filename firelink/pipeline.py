import pickle

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

    def save_fire(self, file, file_type="ember"):
        if file_type == "ember":
            with open(file, "wb") as obj:
                pickle.dump(self, obj)
