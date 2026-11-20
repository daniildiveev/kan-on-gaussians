"""Preregistered settings. Change the output directory for a new study."""

FEATURES = 12
SINGLE_KERNELS, PAIR_KERNELS = 4, 2
MLP_WIDTH = 24
PLANTED = ((0, 1), (4, 5), (8, 9))  # Zero-based feature indices.
DATASETS = ("sparse", "additive")
MODELS = ("mgs_kan", "mlp")
SIZES = (2048, 512, 2048)
DATA_SEEDS = (2311, 2411, 2511)
INIT_SEEDS = (71, 73, 79)
TUNE_DATA_SEED, TUNE_INIT_SEED = 2211, 67
PILOT_DATA_SEED, PILOT_INIT_SEED = 2111, 61
NOISE_SD = 0.10
SELECTION_FRACTION = 0.05
BATCH_SIZE, EPOCHS, PATIENCE, THREADS = 256, 150, 35, 2
LEARNING_RATES = (0.003, 0.01)
# MGS-KAN uses sum of L2 norms over pair amplitude groups. MLP uses Adam L2.
REGULARIZATION = {"mgs_kan": (0.0001, 0.001, 0.005), "mlp": (0.0, 0.0001, 0.001)}
PILOT_SIZE, PILOT_EPOCHS, PILOT_MAX_RMSE = 64, 350, 0.05

# Substitute fixture. The proposal's exact numeric constants were unavailable.
GAUSSIAN_CENTER = (0.35, -0.45)
GAUSSIAN_AMPLITUDE = 1.4
GAUSSIAN_PRECISION = ((3.0, 1.4), (1.4, 1.3))
