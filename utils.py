import random
import numpy as np
from dataset import *
from distutils import util

def str2bool(v):
    return bool(util.strtobool(v))

def setup_seed(random_seed, cudnn_deterministic=True):
    # # initialization
    # torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False


