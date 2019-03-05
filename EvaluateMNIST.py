from Classification import *


class EvaluateMNIST(object):
    def __init__(self, FILE, HIDDEN_UNITS, BATCH_SIZE=125, TEST_BATCH_SIZE=1000, CLASSES=10, INPUT_SIZE=28 * 28):
        LAYERS = np.array([HIDDEN_UNITS, HIDDEN_UNITS])

        # These are not used.
        TRAIN_EPOCHS = 0
        SAMPLES = 1
        PI = 0.
        SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0.)])
        SIGMA_2 = torch.cuda.FloatTensor([math.exp(-8.)])
        LR = 1e-3

        self.mnist = MNIST(BATCH_SIZE=BATCH_SIZE,
                           TEST_BATCH_SIZE=TEST_BATCH_SIZE,
                           CLASSES=CLASSES,
                           TRAIN_EPOCHS=TRAIN_EPOCHS,
                           SAMPLES=SAMPLES,
                           hasScalarMixturePrior=True,
                           PI=PI,
                           SIGMA_1=SIGMA_1,
                           SIGMA_2=SIGMA_2,
                           INPUT_SIZE=INPUT_SIZE,
                           LAYERS=LAYERS,
                           ACTIVATION_FUNCTIONS=np.array(['relu', 'relu', 'softmax']),
                           LR=LR)

        # First load model from file, then initialise
        self.mnist.net.load_state_dict(torch.load(FILE))
        self.mnist.net.eval()


ev = EvaluateMNIST(FILE='Models/BBB_MNIST_400_sample1_lr1e-3_mixed.pth', HIDDEN_UNITS=400)

# ...
# ...
# ... Play around with ev.mnist.net and change weights
# ...
# ...

print(ev.mnist.test(valid=False))