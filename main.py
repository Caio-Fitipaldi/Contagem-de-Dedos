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
    contornos, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Pode ser ajustado de acordo com theresholded

    # Se não achou nada, não tem mão
    if len(contornos) == 0:
        return None
    else:
        # Assume que o maior contorno é a mão
        maoSegmento = max(contornos, key=cv2.contourArea)

        return (thresholded, maoSegmento)

def contDedos(thresholded, segmentoMao):
    convexoMin = cv2.convexHull(segmentoMao)

    # pontos extremos
    top    = tuple(convexoMin[convexoMin[:, :, 1].argmin()][0])
    bottom = tuple(convexoMin[convexoMin[:, :, 1].argmax()][0])
    left   = tuple(convexoMin[convexoMin[:, :, 0].argmin()][0])
    right  = tuple(convexoMin[convexoMin[:, :, 0].argmax()][0])

    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    distancia = pairwise.euclidean_distances([(cX, cY)], [left, right, top, bottom])[0]
    distanciaMax = distancia.max()
    raio = int(0.9 * distanciaMax)
    perimetroCircu = (2 * np.pi * raio)

    boxCircular = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(boxCircular, (cX, cY), raio, 255, 10)

    boxCircular = cv2.bitwise_and(thresholded, thresholded, mask=boxCircular)

    contornos, hierarchy = cv2.findContours(boxCircular.copy(),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)

    cont = 0
    for i in contornos:
        (x, y, w, h) = cv2.boundingRect(i)
        foraMao = (cY + (cY * 0.25)) > (y + h)
        limite = ((perimetroCircu * 0.25) > i.shape[0])
        if foraMao and limite:
            cont += 1

    return cont

cam = cv2.VideoCapture(0)

frames = 0

while True:
    ret, frame = cam.read()

    frameCopia = frame.copy()

    box = frame[boxTop:boxBottom, boxRight:boxLeft]

    cinza = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)

    cinza = cv2.GaussianBlur(cinza, (7,7), 0)

    if frames < 60:
        calcAvg(cinza, pesoAcumulado)

        if frames <= 59:
            cv2.putText(frameCopia, "Aguarde, coletando fundo", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Contagem de Dedos", frameCopia)
    else:

        mao = segmento(cinza)

        if mao is not None:

            thresholded, segmentoMao = mao

            #Desenha box onde fica a contagem de dedos da mão
            cv2.drawContours(frameCopia, [segmentoMao+(boxRight, boxTop)], -1, (255, 0, 0), 5)

            dedos = contDedos(thresholded, segmentoMao)

            cv2.putText(frameCopia, str(dedos), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Theresholded", thresholded)
    cv2.rectangle(frameCopia, (boxLeft, boxTop), (boxRight, boxBottom), (0, 0, 255), 5)

    frames+=1

    cv2.imshow("Contador de Dedos", frameCopia)

    key = cv2.waitKey(1) & 0xFF

    if key == "q":
        break

cam.release()