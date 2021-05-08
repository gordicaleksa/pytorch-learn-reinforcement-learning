import os


BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')
CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'checkpoints')
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')


# Make sure these exist as the rest of the code assumes it
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
