"""
The input variables for prs_analysis training.py

"""

# ---------------
# INPUT VARIABLES

# experiment = 10  # Updated
# Will be ignored if using PRS only (i.e., NG Feature Set is None)
# For those, will use prevalent cases

# -- General
incident_years = 10

elastic_net = False
regression = False
batch_multiples = 1  # This x batch_size is the total "effective" batch size with gradient accumulation
accum_grads = False  # If True, then batch_multiples will be considered
multitask = False
# Only relevant if multivariate regression is True
encoded_multivariate_label = False
mse_loss = False
# -- General Variables
penalize_class = True
num_workers = 0

# Use the lowest common "denominator" sample set. Individuals with available genotype
# data, with available medical history data, excluding those removed from UK Biobank
# Include only those with British ancestry to avoid potential biases
# If the disease is females only, make a note of that
sample_subset = True
# gpu_number = 0  # The GPU number to use on the system
train_pct = 0.7  # Percent of the dataset used for training
valid_pct = 0.2  # Percent of the dataset for validation
test_pct = 0.1  # Percent of the dataset for testing
train_batch_size = 256
valid_batch_size = 1024

pretrained_NG2 = False  # Loads a pre-trained LinearVAE model and appends the ffnn model
fixed_pretrained_weights = True  # If not True, then the model allows for fine-tuning rather than just pretraining
use_embedding = True  # If this is True, the script will the pretrained embedding dataset as fixed
balance_train = False
balance_val = False
batch_normalize = True
num_trials = 10  # TODO change back later
