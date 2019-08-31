import sys
from minisom import MiniSom

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

maxEpochs = 100
somSize = 10
__dir='./datasets/'
filename='solo.csv'
try:
    arg = sys.argv[1:]
    
    if(arg[0]):
        maxEpochs = int(arg[0])
    if(arg[1]):
        somSize = int(arg[1])
    if(arg[2]):
        filename = arg[2]
except :
    pass





data = np.genfromtxt(__dir+filename, delimiter=',')
# Elimine header
data = data[1:]
dataColuns = 1
try:
    dataColuns = len(data[0])
except:
    pass
print('Linhas: ',len(data), ' Colunas: ',dataColuns)


data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 0
, data)


# Initialization and training
som = MiniSom(somSize, somSize, dataColuns, sigma=2.5, learning_rate=0.5, 
              neighborhood_function='gaussian', random_seed=4)

som.pca_weights_init(data)
print("Training...")
som.train_batch(data, maxEpochs, verbose=True)  # random training
print("\n...ready!")



plt.figure(figsize=(7, 7))
# Plotting the response for each pattern in the iris dataset

plt.subplot(2,1,1)
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # plotting the distance map as background
plt.title('Matriz - U')
plt.colorbar()

plt.subplot(2,1,2)
frequencies = np.zeros((somSize, somSize))
for position, values in som.win_map(data).items():
    frequencies[position[0], position[1]] = len(values)
plt.title('Densidade de Dados')
plt.pcolor(frequencies)
plt.colorbar()
plt.show()