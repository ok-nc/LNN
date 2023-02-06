"""
Parameter file for specifying the running parameters for forward Lorentz Neural Network (LNN) model
"""
# Model Architectural Parameters
NUM_LORENTZ_OSC = 4         # Defines number of Lorentzian oscillators to use for each eps and mu
LINEAR = [4, 100,250,250,100]   # Defines dimensions of fully-connected hidden layers

# Optimization parameters
OPTIM = "Adam"      # Specifies which optimizer to use (Adam is default)
REG_SCALE = 1e-4      # Regularization scale; controls weight decay during gradient descent
BATCH_SIZE = 128       # Batch size of single train/test batch
EVAL_STEP = 10         # Number of epochs between model evaluations
RECORD_STEP = 10        # Number of epochs between plotting/graphing intermediate results
TRAIN_STEP = 30000      # Total number of training epochs
LEARN_RATE = 1e-2       # Learning rate; determines magnitude of weight updates during gradient descent
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5     # Learning rate decay constant; controls magnitude of LR reduction during training
STOP_THRESHOLD = 1e-5   # Stopping point of training, if error falls below this value
USE_CLIP = False        # Flag for using gradient clipping, to control exploding gradients
GRAD_CLIP = 500          # Maximum gradient value, if grad clipping is enabled
USE_WARM_RESTART = True     # Flag for using warm restarts; WR resets LR to initial value in defined periods
LR_WARM_RESTART = 200       # Warm restart period; LR resets to initial value every X number of epochs
LOSS_FACTOR = 5000          # Hyperparameter for tuning amplitude of loss term
NTWK_NOISE = 0.01           # Adds gaussian noise to network weights (may or may not help training -- STILL TESTING)

# Data Specific parameters
X_RANGE = [i for i in range(0, 4)]      # Defines number of "feature" or input variables in training data files
Y_RANGE = [i for i in range(1, 1001,2)]     # Defines number of "label" or spectral points in training data files
FREQ_LOW = 20.02            # Defines lower frequency bound (mostly for plotting functions)
FREQ_HIGH = 40            # Defines upper frequency bound (mostly for plotting functions)
NUM_SPEC_POINTS = 500            # Defines number of spectral points; must be consistent with Y_range size
FORCE_RUN = True
DATA_DIR = ''                # For local directory usage
# DATA_DIR = 'C:/Users/labuser/DL_AEM/'                # For Valkyrie office desktop usage
GEOBOUNDARY =[1.3, 0.975, 6, 37, 2.0, 3, 7, 43.749]     # Format for geoboundary is [p0_min... pf_min p0_max... pf_max]
# GEOBOUNDARY =[1.3, 0.975, 6, 34.539, 2.4, 3, 7, 43.749]  # Original geoboundary for ADM cylinder dataset (lower loss)
NORMALIZE_INPUT = True          # Flag for normalizing input from [-1, 1] using geoboundary
TEST_RATIO = 0.2                # Train/test split ratio
DATA_REDUCE = 0                 # Sets maximum size for training dataset; Default value of 0 means no data reduction

# Running specific
USE_CPU_ONLY = False            # Turns off GPU usage
MODEL_NAME  = None              # Save name for model (default is timestamp)
EVAL_MODEL = "Eval model"      # Name of model folder to evaluate
# EVAL_MODEL = "20210918_181306_paper"
# EVAL_MODEL = "20210806_101530_eval"
NUM_PLOT_COMPARE = 10           # Defines number of sample plots to monitor in tensorboard during training

# Dummy variables               # These dummy variables are for the DNN model, but are included here
USE_CONV = False                # for code consistency (to easily switch between DNN and LNN)
CONV_OUT_CHANNEL = [4, 4, 4]
CONV_KERNEL_SIZE = [8, 5, 5]
CONV_STRIDE = [2, 1, 1]
