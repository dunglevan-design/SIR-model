# SIR-model
SIR model remake
A simple implementation of the SIR model. 

  dS/dt   = -beta * S * I
  dI/dt   = beta  * S * I  - gamma* I
  dR/dt   = gamma * I  
  where S = susceptibles, I = infectives, R = Removed
The program aims at producing an estimate of beta and gamma that minimizes the mean squared error from the real data.
If the model sucessfully found [beta,gamma], A graph of the model would be produced and saved.
