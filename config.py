import transformers, torch


MAX_SEQUENCE_LENGTH = 2048
NUM_EPOCHS = 400
TRANING_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 8
TEST_BATCH_SIZE = 12
LEARNING_RATE = 3e-6
RANDOM_SEED = 666
NUM_WORKERS = 4
NUM_WARMUP_STEPS = 2e3
NUM_TRAINING_STEPS = 1e6
TRANING_LEANGTH = 100
VALIDATION_LEANGTH = 100
TEST_LEANGTH = 100
NUM_ACCUM_STEPS = 8
CHECK_LOSS_AFTER_NUM_STEP = 10
SAVE_AFTER_NUM_STEP = 10
USE_AMP = True
L2_PENALTY = 0.01
MAX_GRAD_NORM = 1.0

DATASET_NAME = 'v3xlrm1nOwo1/AnimeSongsLyrics'
OUTPUT_DIR = 'AkiyamaMio' # akiyama mio

CHECKPOINT = 'lightblue/karasu-1.1B'
ACCESS_TOKEN = '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'

TOKENIZER = transformers.AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_MODEL_PATH = 'mio.bin'

