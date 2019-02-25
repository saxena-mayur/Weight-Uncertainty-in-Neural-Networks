from BayesBackpropagation import *

def train(net, optimizer, data, target, NUM_BATCHES):
    net.train()
    for i in range(NUM_BATCHES):
        net.zero_grad()
        x = data[i].reshape((-1, 1))
        y = target[i].reshape((-1,1))
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(x,y)
        loss.backward()
        optimizer.step()

TRAIN_EPOCHS = 500
SAMPLES = 1
TEST_SAMPLES = 10
BATCH_SIZE = 100
NUM_BATCHES = 5
TEST_BATCH_SIZE = 50
CLASSES = 1

print('Generating Data set.')

Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))

x = np.random.uniform(-0.1, 0.45, size=(NUM_BATCHES,BATCH_SIZE))
noise = np.random.normal(0, 0.02, size=(NUM_BATCHES,BATCH_SIZE))
y = x + 0.3*np.sin(2*np.pi*(x+noise)) + 0.3*np.sin(4*np.pi*(x+noise)) + noise
X = Var(x)
Y = Var(y)

x_test = np.linspace(-0.5, 1,TEST_BATCH_SIZE)
y_test = x_test + 0.3*np.sin(2*np.pi*x_test) + 0.3*np.sin(4*np.pi*x_test)
X_test = Var(x_test)

#plt.scatter(x, y, c='navy', label='target')
#plt.legend()
#plt.tight_layout()
#plt.show()
print(x.shape)

#Training
PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])

print('Training Begins!')
net = BayesianNetwork(inputSize = 1,\
                      CLASSES = CLASSES, \
                      layers=np.array([100,400,400]), \
                      activations = np.array(['relu','relu','relu','none']), \
                      SAMPLES = SAMPLES, \
                      BATCH_SIZE = BATCH_SIZE,\
                      NUM_BATCHES = NUM_BATCHES,\
                      hasScalarMixturePrior = True,\
                      pi = PI,\
                      sigma1 = SIGMA_1,\
                      sigma2 = SIGMA_2).to(DEVICE)
optimizer = optim.Adam(net.parameters())
for epoch in range(TRAIN_EPOCHS):
    train(net, optimizer,data=X,target=Y,NUM_BATCHES=NUM_BATCHES)

print('Training Ends!')

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
plt.savefig('./Results/Regression.png')
#plt.show()