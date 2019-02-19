from BayesBackpropagation import *

def train(net, optimizer, data, target):
    net.train()
    net.zero_grad()
    loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
    loss.backward()
    optimizer.step()

TRAIN_EPOCHS = 100
SAMPLES = 1
TEST_SAMPLES = 10
BATCH_SIZE = 200
NUM_BATCHES = 1
TEST_BATCH_SIZE = 300
CLASSES = 1

Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))

x = np.random.uniform(-1, 1, size=BATCH_SIZE).reshape((-1, 1))
noise = np.random.normal(0, 0.02, size=BATCH_SIZE).reshape((-1, 1))
#y = x ** 2
y = x + 0.3*np.sin(2*np.pi*(x+noise)) + 0.3*np.sin(4*np.pi*(x+noise)) + noise
X = Var(x)
Y = Var(y)

x_test = np.linspace(-2, 2,TEST_BATCH_SIZE)
#y_test = x_test ** 2
y_test = x_test + 0.3*np.sin(2*np.pi*x_test) + 0.3*np.sin(4*np.pi*x_test)
X_test = Var(x_test)

plt.scatter(x, y, c='navy', label='target')
plt.legend()
plt.tight_layout()
plt.show()

#Training
net = BayesianNetwork(inputSize = 1,\
                      CLASSES = CLASSES, \
                      layers=np.array([100,400,100]), \
                      activations = np.array(['relu','relu','relu','none']), \
                      SAMPLES = SAMPLES, \
                      BATCH_SIZE = BATCH_SIZE,\
                      NUM_BATCHES = NUM_BATCHES).to(DEVICE)
optimizer = optim.Adam(net.parameters())
for epoch in range(TRAIN_EPOCHS):
    train(net, optimizer,data=X,target=Y)

#Testing
outputs = torch.zeros(TEST_SAMPLES+1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
for i in range(TEST_SAMPLES):
    outputs[i] = net.forward(X_test)
outputs[TEST_SAMPLES] = net.forward(X_test)
pred_mean = outputs.mean(0).data.numpy().squeeze(1)
pred_std = outputs.std(0).data.numpy().squeeze(1)

plt.scatter(x, y, c='navy', label='target')

plt.plot(x_test, pred_mean, c='royalblue', label='Prediction')
plt.fill_between(x_test, pred_mean - 3 * pred_std, pred_mean + 3 * pred_std,
                     color='cornflowerblue', alpha=.5, label='+/- 3 std')

plt.plot(x_test, y_test, c='grey', label='truth')

plt.legend()
plt.tight_layout()
plt.show()