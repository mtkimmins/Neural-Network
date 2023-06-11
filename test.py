import numpy as np


a = np.array([0,0,0,1]).reshape((4,1))
b = np.array([0,0,0,1]).reshape((4,1))
c = np.array([0,1,0,0]).reshape((4,1))



d = np.argmax(a)
e = np.argmax(b)
f = np.argmax(c)

print(a==b)
print(a==c)
print(d)
print(e)
print(f)
print(d.item())
print(e.item())
print(f.item())
print(d.item()==e.item())
print(d.item()==f.item())