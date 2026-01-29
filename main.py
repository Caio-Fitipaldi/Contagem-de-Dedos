#Importando Bibliotecas necessárias
import cv2
import numpy as np

from sklearn.metrics import pairwise

# Modelo de fundo da cena
fundo = None # Inicialmente não há cena inicializada

# Peso(Alpha), controle de atualização do fundo
pesoAcumulado = 0.4

# Região de detecção
boxTop = 20
boxBottom = 300
boxRight = 300
boxLeft = 600

# Função ue atualiza o fundo de cena
def calcAvg(frame, pesoAcumulado):
    global fundo
    if fundo is None: # Fundo não inicializado
        fundo = frame.copy().astype("float")
        return None
    # Atualização do fundo de acordo com  formula
    cv2.accumulateWeighted(frame, fundo, pesoAcumulado)

