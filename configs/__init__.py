from .config import banet

class cfg_dict(object):

        def __init__(self, d):
                    self.__dict__ = d

cfg = cfg_dict(banet)
