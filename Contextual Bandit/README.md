# Deep Contextual Bandit Experiment

## Mushroom Case Stuty: 
We are provided with a list of 8124 mushrooms, each having 22 features (characteristcs of the mushroom) and 1 label (poisonous or edible). Our agent can carry out 2 actions: eat a mushroom or not eat a mushroom. The problem context is the vector of features which is associated with the mushroom which the agent is about to eat/non eat. If our agent eats an edible mushroom, it receives a reward of 5. If the agent eats a poisonous mushroom, it receives a reward of -35 with 0.5 probability and a reward of 5 with 0.5 probaility. If the agent doesn't eat, it receives a reward of 0.

We are also provided an oracle. The oracle always selects the right action, receiving a reward of 5 when it eats an edible mushroom, and a reward of 0 when it doesn't eat. The objective is to minimise the total cumulative regret of the agents.


