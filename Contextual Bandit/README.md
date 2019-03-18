# Deep Contextual Bandit Experiment

## Mushroom Case Stuty: 
We are provided with a list of 8124 mushrooms, each having 22 features (characteristcs of the mushroom) and 1 label (poisonous or edible). Our agent can carry out 2 actions: eat a mushroom or not eat a mushroom. The problem context is the vector of features which is associated with the mushroom which the agent is about to eat/non eat. If our agent eats an edible mushroom, it receives a reward of 5. If the agent eats a poisonous mushroom, it receives a reward of -35 with 0.5 probability and a reward of 5 with 0.5 probaility. If the agent doesn't eat, it receives a reward of 0.

We are also provided an oracle. The oracle always selects the right action, receiving a reward of 5 when it eats an edible mushroom, and a reward of 0 when it doesn't eat. The objective is to minimise the total cumulative regret of the agents.

## Implementations
We have a Pytorch and Tensorflow implementation. The Pytorch implementation achieves exact results for the greedy agents. Experiments for the BBB agent are currently running, but initial results were not as expected. The tensorflow implementation is taken from the Tensorflow Deep Contextual Bandits repository, I have mantained only the code which is relevant to our experiment. Works, but needs to be tailored to our experiment.

## Experiments
Run experiments for the pytorch implementation varying learning rates and optimisers. 
For Greedy agents: tried SGD and Adam with lr = 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, achieving best results with SGD, lr = 0.0005. I have saved the data and plot for this experiment in the Results folder.
For BBB agent: tried SGD and Adam with lr = 0.01, 0.05, 0.001, using number of samples: 2, 5, 10.


## Experiment (Guillaume)

Pytorch (guillaume) implementation,
learning rate 0.001, bbb_number of_samples 2, for greedy agent and bbb_net, for 10 000 steps, starting with an empty buffer, with proba clipping to avoid numerical problems

loss: sum of square distances in minibatch. Maybe take a smaller lr then ?

Step 2626/10000, Edible:1,  BBB : 14000, Greedy: 6405

A2: lr=2e-5, (but sum of square distance ->*64)
to make Bayes converge, it is also possible to decrease the weight of kl.
maybe use exponential reweighting ? that is strange.





