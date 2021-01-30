# demo of linear least squares
import numpy as np

lrs = np.array([
    [.460, -.059],
    [.575, -0.076],
    [.566, .080],
    [.581, 0.099],
    [.427, 0.111],
    [.046, .684],
    [-.003, .580],
    [.022, .625],
    [-.155, .646],
    [-.207, .857]
])

q = 8.0
vx = np.array([
   -.002,
   .020,
   .058,
   .066,
   .094,
   .066,
   .024,
   .042,
   .021,
   .041
])

(x, residuals, rank, s)= np.linalg.lstsq(lrs, q*vx)
print(x)
print(residuals, rank, s)
