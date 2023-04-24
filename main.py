import networkParts as nwp
import mathLib as ml

net = nwp.Network(2, [2], 1, ml.Sigmoid, 0.1)

inputs = [
    [0,0],
    [1,1],
    [1,0],
    [0,1]
]

answers = [
    [0],
    [0],
    [1],
    [1]
]

# print("INITIAL")
# # net.print()
# net.feedforward(inputs[1])
# print("AFTER FEEDFORWARD")
# # net.print()
# net.backpropagate(answers[1])
# print("AFTER BACKPROP")
# net.print()
net.train(inputs, answers, 100000)
for n in inputs:
    net.predict(n)