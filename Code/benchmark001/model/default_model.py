from ..config import get_spa
from flgo.utils.fmodule import FModule

class Model(FModule):
    def __init__(self, in_size=1703, hid_size=128, out_size=5, dropout = 0.1, batch_size = 128, num_trans =2):
        super().__init__()
        self.model = get_spa(in_size, hid_size, out_size, 2)
        if hasattr(self.model, 'compute_loss'):
            self.compute_loss = self.model.compute_loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)