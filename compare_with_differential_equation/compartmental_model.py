import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

# # Parameters
# N = 1000
# beta = 0.5       # Transmission rate
# gamma = 1 / 10   # Recovery rate
# sigma = 1 / 5

# Parameters
N = 1000
beta = 20       # Transmission rate
gamma = 3   # Recovery rate
sigma = 3.5#1 / 50

# Initial conditions
I0 = int(0.01 * N)  # Initially infectious
E0 = 0              # No one exposed initially
S0 = N - I0 - E0    # Rest are susceptible
R0 = 0              # No one recovered initially

y0 = [S0, E0, I0, R0]

# Time vector (100 days)
days = 2
#t = np.linspace(0, days, days + 1)
t = np.linspace(0, days, 100)
# SEIR model
def seir(y, t, beta, sigma, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Solve the ODE system
result = odeint(seir, y0, t, args=(beta, sigma, gamma, N))
S_total, E_total, I_total, R_total = result.T

# Save to CSV
df = pd.DataFrame({
    'Day': t,
    'Susceptible': S_total,
    'Exposed': E_total,
    'Infectious': I_total,
    'Recovered': R_total
})
df.to_csv('seir_output.csv', index=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, S_total, label='Susceptible')
plt.plot(t, E_total, label='Exposed')
plt.plot(t, I_total, label='Infectious')
plt.plot(t, R_total, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of individuals')
plt.title('Simple SEIR Model')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
