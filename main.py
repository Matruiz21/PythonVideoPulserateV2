import cv2
import mediapipe as mp
import time
import numpy as np
from scipy import signal
import os
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
video_path = "D:/User/Nerdex/Documentos/ITBA/Tesis/Videos/source_videos_part_16-001/source_videos/W135/BlendShape/camera_front/W135_BlendShape_camera_front.mp4"

# Inicializamos MediaPipe para la detección de rostros
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Parámetros de detección de rostros
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Captura de video (puede ser desde la cámara o un archivo de video)
cap = cv2.VideoCapture(video_path)  # O usar 0 para cámara

# Parámetros del análisis de pulsos
fs = 30  # Frecuencia de muestreo (FPS)
window = 300  # Número de muestras para cada medición
skin_vec = [0.3841, 0.5121, 0.7682]  # Vector de piel
B, G, R = 0, 1, 2

mean_colors = []
timestamps = []
mean_colors_resampled = np.zeros((3, 1))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Fin del video o error")
        break

    # Convertimos el fotograma a RGB (MediaPipe trabaja en RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizamos la detección de rostros
    results = face_detection.process(frame_rgb)

    # Si se detectan rostros
    if results.detections:
        for detection in results.detections:
            # Dibujar las cajas y puntos en el rostro detectado
            mp_drawing.draw_detection(frame, detection)

            # Obtener las coordenadas de la caja de la cara
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Extraer la región de interés (rostro)
            face = frame[y:y + h, x:x + w]

            if face.size > 0:
                # Calcular el color promedio del rostro
                mean_colors.append(face.mean(axis=0).mean(axis=0))
                timestamps.append(time.time())

                # Resamplear los colores
                t = np.arange(timestamps[0], timestamps[-1], 1 / fs)
                mean_colors_resampled = np.zeros((3, t.shape[0]))

                for color in [B, G, R]:
                    resampled = np.interp(t, timestamps, np.array(mean_colors)[:, color])
                    mean_colors_resampled[color] = resampled

                # Aplicar el método de crominancia si tenemos suficientes muestras
                if mean_colors_resampled.shape[1] > window:
                    col_c = np.zeros((3, window))

                    for col in [B, G, R]:
                        col_stride = mean_colors_resampled[col, -window:]  # Seleccionar las últimas muestras
                        y_ACDC = signal.detrend(col_stride / np.mean(col_stride))
                        col_c[col] = y_ACDC * skin_vec[col]

                    X_chrom = col_c[R] - col_c[G]
                    Y_chrom = col_c[R] + col_c[G] - 2 * col_c[B]
                    Xf = utils.bandpass_filter(X_chrom)  # Función de filtrado
                    Yf = utils.bandpass_filter(Y_chrom)
                    Nx = np.std(Xf)
                    Ny = np.std(Yf)
                    alpha_CHROM = Nx / Ny

                    x_stride = Xf - alpha_CHROM * Yf
                    amplitude = np.abs(np.fft.fft(x_stride, window)[:int(window / 2 + 1)])
                    normalized_amplitude = amplitude / amplitude.max()  # Amplitud normalizada

                    frequencies = np.linspace(0, fs / 2, int(window / 2) + 1) * 60
                    bpm_index = np.argmax(normalized_amplitude)
                    bpm = frequencies[bpm_index]
                    snr = utils.calculateSNR(normalized_amplitude, bpm_index)  # Calcular la relación señal/ruido

                    # Mostrar el BPM y SNR en el fotograma
                    utils.put_snr_bpm_onframe(bpm, snr, frame)

    # Mostrar el video con las detecciones
    cv2.imshow('Video', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()