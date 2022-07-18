import interferometer as itf
import numpy as np
import random

M = 10
I = itf.Interferometer()
angles = []
phases = []

for depth in range(10):

    if depth % 2 == 0:
        shift = 0
    else:
        shift = 1

    for i in range(M // 2 - shift):
        j = 2*i + shift + 1  # Interferometer mode index starts from 1
        phase = 0.5 * random.random() * np.pi
        angle = 0.5 * random.random() * np.pi

        angles.append(angle)
        phases.append(phase)

        bs = itf.Beamsplitter(j, j+1, angle, phase)

        I.add_BS(bs)

I.draw()

U = I.calculate_transformation()
print(U)

np.savetxt('testU', U, delimiter=',')