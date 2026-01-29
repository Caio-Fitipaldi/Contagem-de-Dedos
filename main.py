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


def segmento(frame, thresh_value=25):
    # Diferença entre fundo e frame atual
    diff = cv2.absdiff(fundo.astype("uint8"), frame)

    # limiarização (threshold) para gerar imagem binária
    ret, thresholded = cv2.threshold(diff, thresh_value, 255, cv2.THRESH_BINARY)

    # Encontrar contornos (regiões brancas → possíveis mãos)
    imagem, contornos, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Pode ser ajustado de acordo com theresholded

    # Se não achou nada, não tem mão
    if len(contornos) == 0:
        return None
    else:
        # Assume que o maior contorno é a mão
        maoSegmento = max(contornos, key=cv2.contourArea)

        return (thresholded, maoSegmento)
