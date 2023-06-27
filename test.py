import numpy as np
import NeuralNetwork as nn


net = nn.Network([10,5,5,3])

data = np.array([[2],[2],[2],[2],[2],[2],[2],[2],[2],[2]]).reshape((10,1))
label = np.array([[0],[0],[0]]).reshape((3,1))

'''''
2,2,2,2,2,2,2,2,2,2

(2*10+1),(2*10+1),(2*10+1),(2*10+1),(2*10+1)
21,21,21,21,21
1,1,1,1,1

(1*5+1),(1*5+1),(1*5+1),(1*5+1),(1*5+1),
6,6,6,6,6
1,1,1,1,1

(1*5+1),(1*5+1),(1*5+1)
6,6,6
1,1,1
'''''
net.feedforward(data)
# for n in net.pre_activations:
#     print(n)

'''
1,1,1

(1*3),(1*3),(1*3),(1*3),(1*3)
3,3,3,3,3

(3*5),(3*5),(3*5),(3*5),(3*5),
15,15,15,15,15
========
wt
dx = y*(1-y)
1,1,1

0,0,0



[][][][][]
[][][][][]
[][][][][]





'''

errors = net.backpropagate(label)
dw,db,er = errors
for n in db:
    print(n)