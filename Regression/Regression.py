import math
import torch.optim as optim
sys.path.append('../')
from BayesBackpropagation import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Define training step for regression
def train(net, optimizer, data, target, NUM_BATCHES):
    #net.train()
    for i in range(NUM_BATCHES):
        net.zero_grad()
        x = data[i].reshape((-1, 1))
        y = target[i].reshape((-1,1))
        loss = net.BBB_loss(x, y)
        loss.backward()
        optimizer.step()

#Hyperparameter setting
TRAIN_EPOCHS = 900
SAMPLES = 5
TEST_SAMPLES = 10
BATCH_SIZE = 200
NUM_BATCHES = 10
TEST_BATCH_SIZE = 50
CLASSES = 1
PI = 0.25
SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(DEVICE)
SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(DEVICE)

print('Generating Data set.')

#Data Generation step
if torch.cuda.is_available():
    Var = lambda x, dtype=torch.cuda.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor
else:
    Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor

x = np.random.uniform(-0.1, 0.61, size=(NUM_BATCHES,BATCH_SIZE))
noise = np.random.normal(0, 0.02, size=(NUM_BATCHES,BATCH_SIZE)) #metric as mentioned in the paper
y = x + 0.3*np.sin(2*np.pi*(x+noise)) + 0.3*np.sin(4*np.pi*(x+noise)) + noise

x_test = np.linspace(-0.5, 1,TEST_BATCH_SIZE)
y_test = x_test + 0.3*np.sin(2*np.pi*x_test) + 0.3*np.sin(4*np.pi*x_test)

def BBB_Regression(x,y,x_test,y_test):

    print('BBB Training Begins!')

    X = Var(x)
    Y = Var(y)
    X_test = Var(x_test)

    #Declare Network
    net = BayesianNetwork(inputSize = 1,\
                        CLASSES = CLASSES, \
                        layers=np.array([16,16,16]), \
                        activations = np.array(['relu','relu','relu','none']), \
                        SAMPLES = SAMPLES, \
                        BATCH_SIZE = BATCH_SIZE,\
                        NUM_BATCHES = NUM_BATCHES,\
                        hasScalarMixturePrior = True,\
                        PI = PI,\
                        SIGMA_1 = SIGMA_1,\
                        SIGMA_2 = SIGMA_2,\
                        GOOGLE_INIT= False).to(DEVICE)

    #Declare the optimizer
    optimizer = optim.SGD(net.parameters(),lr=1e-3,momentum=0.95)

    for epoch in range(TRAIN_EPOCHS):
        if (epoch)%10 == 0:
            print('Epoch: ', epoch)
        train(net, optimizer,data=X,target=Y,NUM_BATCHES=NUM_BATCHES)

    print('Training Ends!')

    #Testing
    outputs = torch.zeros(TEST_SAMPLES+1, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
    for i in range(TEST_SAMPLES):
        outputs[i] = net.forward(X_test)
    outputs[TEST_SAMPLES] = net.forward(X_test)
    pred_mean = outputs.mean(0).data.cpu().numpy().squeeze(1) #Compute mean prediction
    pred_std = outputs.std(0).data.cpu().numpy().squeeze(1) #Compute standard deviation of prediction for each data point

    #Visualization
    plt.fill_between(x_test, pred_mean - 3 * pred_std, pred_mean + 3 * pred_std,
                        color='cornflowerblue', alpha=.5, label='+/- 3 std')
    plt.scatter(x, y,marker='x', c='black', label='target')
    plt.plot(x_test, pred_mean, c='red', label='Prediction')
    plt.plot(x_test, y_test, c='grey', label='truth')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../Results/Regression_BBB.png')
    plt.savefig('../Results/Regression_BBB.eps', format='eps', dpi=1000)
    plt.clf()

    #Save the trained model
    torch.save(net.state_dict(), './Regression.pth')

#Comparing to standard neural network
def NN_Regression(x,y,x_test,y_test):

    print('SGD Training Begins!')

    x = x.flatten()
    X = Var(x)
    X = torch.unsqueeze(X,1)
    
    y = y.flatten()
    Y = Var(y)
    Y = torch.unsqueeze(Y,1)
    X_test = Var(x_test)
    X_test = torch.unsqueeze(X_test,1)

    class Net(torch.nn.Module):
        def __init__(self, n_feature, n_hidden, n_output):
            super(Net, self).__init__()
            self.l1 = torch.nn.Linear(n_feature, n_hidden[0])   # hidden layer 1
            self.l2 =  torch.nn.Linear(n_hidden[0], n_hidden[1])   # hidden layer 2
            self.l3 =  torch.nn.Linear(n_hidden[1], n_hidden[2])   # hidden layer 3
            self.predict = torch.nn.Linear(n_hidden[2], n_output)   # output layer

        def forward(self, x):
            x = F.relu(self.l1(x))      # activation function for hidden layer 1
            x = F.relu(self.l2(x))      # activation function for hidden layer 2
            x = F.relu(self.l3(x))      # activation function for hidden layer 3
            x = self.predict(x)         # linear output
            return x

    net = Net(n_feature=1, n_hidden=[16,16,16], n_output=1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    for epoch in range(2000):
        if (epoch)%10 == 0:
            print('Epoch: ', epoch)
        prediction = net(X)     # input x and predict based on x
        loss = loss_func(prediction, Y)     # must be (1. nn output, 2. target)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    
    prediction = net(X_test)
    
    #Visualization
    plt.scatter(x, y,marker='x', c='black', label='target')
    plt.plot(x_test, prediction.detach().numpy(), c='red', label='Prediction')
    plt.plot(x_test, y_test, c='grey', label='truth')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../Results/Regression_NN.png')
    plt.savefig('../Results/Regression_NN.eps', format='eps', dpi=1000)
    plt.clf()

BBB_Regression(x,y,x_test,y_test)
NN_Regression(x,y,x_test,y_test)