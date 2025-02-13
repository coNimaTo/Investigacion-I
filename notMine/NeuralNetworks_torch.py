# Aprendizaje supervisado en redes multicapa
#
# date: 11/02/2025  
# File: MeuralNetworks_introduction.ipynb  
# Author : Pablo Naim Chehade   
# Email: pablo.chehade.villalba@gmail.com  
# GitHub: https://github.com/Lupama2  

# Este código resuelve el mismo problema que el comentado en NeuralNetworks_introduction.ipynb,
# con la diferencia de que se emplea la librería torch


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Definimos la red neuronal
class XORNetwork(nn.Module):
    def __init__(self):
        super(XORNetwork, self).__init__()
        self.hidden = nn.Linear(2, 2)  # Capa oculta con 2 neuronas
        self.output = nn.Linear(2, 1)  # Capa de salida con 1 neurona
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.activation(self.output(x))
        return x

# Definimos la función XOR y generamos los datos
def generate_xor_data():
    x_data = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=torch.float32)
    y_data = torch.tensor([[1], [-1], [-1], [1]], dtype=torch.float32)
    return x_data, y_data

# Definimos la función de entrenamiento usando todos los ejemplos en la actualización de pesos
def train_model(model, x_data, y_data, epochs=2000, lr=0.1):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history = []
    validation_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_data)  # Forward pass con todo el batch
        loss = criterion(y_pred, y_data)
        loss.backward()
        optimizer.step()  # Actualiza los pesos usando todos los ejemplos
        loss_history.append(loss.item())
        
        # Validación en cada epoch
        validation_error = validate_model(model, x_data, y_data)
        validation_history.append(validation_error)
        
        if epoch % 500 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Validation Error: {validation_error:.4f}')
    
    return loss_history, validation_history

# Función de validación
def validate_model(model, x_data, y_data):
    error = 0
    with torch.no_grad():
        y_pred = model(x_data)
        y_pred = torch.where(y_pred >= 0, torch.tensor(1.0), torch.tensor(-1.0))
        error = torch.abs(y_data - y_pred).sum().item() / 2
    
    validation_error = error / len(y_data)
    return validation_error

# Main
if __name__ == "__main__":
    # Generamos los datos
    x_data, y_data = generate_xor_data()
    
    # Creamos la red neuronal
    model = XORNetwork()
    
    # Entrenamos el modelo
    loss_history, validation_history = train_model(model, x_data, y_data)
    
    # Graficamos la evolución del error
    fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    fig.subplots_adjust(hspace=0.02)
    
    epochs = len(loss_history)
    epoch_range = np.arange(epochs)
    
    loss_mean = np.array(loss_history)
    loss_std = np.zeros_like(loss_mean)  # No tenemos múltiples corridas para calcular std
    validation_mean = np.array(validation_history)
    validation_std = np.zeros_like(validation_mean)
    
    ax[0].plot(loss_mean, color="tab:blue")
    ax[0].fill_between(epoch_range, loss_mean - loss_std, loss_mean + loss_std, alpha=0.2, color="tab:blue")
    ax[1].plot(validation_mean, color="tab:blue")
    ax[1].fill_between(epoch_range, validation_mean - validation_std, validation_mean + validation_std, alpha=0.2, color="tab:blue")
    
    # Decoración
    ax[1].set_xlabel("Epoch")
    ax[0].set_ylabel("Error de\nentrenamiento")
    ax[1].set_ylabel("Error de\nprecisión")
    
    ax[0].grid()
    ax[1].grid()
    
    ax[0].set_ylim([0, np.max(loss_mean)])
    ax[1].set_ylim([0, 0.5])
    
    plt.show()
