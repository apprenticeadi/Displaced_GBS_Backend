from src.gbs_experiment import PureGBS

M = 20
depth =5
I = PureGBS.random_interferometer(M, depth = depth)

I.draw()

U = I.calculate_transformation()
print(U)

# np.savetxt('testU', U, delimiter=',')