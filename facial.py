import cv2
import time
import os
import numpy as np
import csv
from datetime import datetime

os.makedirs("videos", exist_ok=True)
os.makedirs("fotos", exist_ok=True)

csv_file = "log_rosto.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "video", "id_rosto", "foto"])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

video_count = 1
gravando = False
start_time = None
out = None
rosto_presente = False
ultima_gravacao = 0
maiores_rostos = {}
detecção_ativa = False

next_id = 1
rosto_ids = {}       
rastreadores = {}     
id_count = {}

def log_rosto(video_name, id_rosto, foto_name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, video_name, id_rosto, foto_name])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ids_para_remover = []
    for id_rosto, rast in rastreadores.items():
        ok, bbox = rast.update(frame)
        if ok:
            x, y, w, h = [int(v) for v in bbox]
            rosto_ids[id_rosto] = (x, y, w, h)
        else:
            ids_para_remover.append(id_rosto)  

    for id_rosto in ids_para_remover:
        del rastreadores[id_rosto]
        del rosto_ids[id_rosto]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        encontrado = False
        cx, cy = x + w//2, y + h//2
        for id_rosto, (rx, ry, rw, rh) in rosto_ids.items():
            rcx, rcy = rx + rw//2, ry + rh//2
            dist = np.hypot(cx - rcx, cy - rcy)
            if dist < 50:
                encontrado = True
                break
        if not encontrado:
            rast = cv2.legacy.TrackerCSRT_create()
            rast.init(frame, (x, y, w, h))
            rastreadores[next_id] = rast
            rosto_ids[next_id] = (x, y, w, h)
            next_id += 1

    if detecção_ativa and len(rosto_ids) > 0:
        if not gravando and not rosto_presente and (time.time() - ultima_gravacao >= 4):
            video_filename = os.path.join("videos", f"video{video_count}.mp4")
            out = cv2.VideoWriter(
                video_filename,
                cv2.VideoWriter_fourcc(*'mp4v'),
                20,
                (frame_width, frame_height)
            )
            gravando = True
            start_time = time.time()
            maiores_rostos = {}
            print(f"➡ Gravando vídeo {video_filename}...")

        rosto_presente = True
    else:
        rosto_presente = False

    if gravando:
        out.write(frame)
        tempo_passado = time.time() - start_time
        tempo_restante = max(0, 3 - tempo_passado)

        for id_rosto, (x, y, w, h) in rosto_ids.items():
            area = w * h
            if id_rosto not in maiores_rostos or area > maiores_rostos[id_rosto][0]:
                maiores_rostos[id_rosto] = (area, frame[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {id_rosto}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"Gravando... {tempo_restante:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if tempo_passado >= 3:
            out.release()
            print(f"✅ Gravação finalizada: {video_filename}")

            for id_rosto, (area, rosto_cortado) in maiores_rostos.items():
                foto_filename = os.path.join("fotos", f"foto{video_count}_ID{id_rosto}.jpg")
                cv2.imwrite(foto_filename, rosto_cortado)
                id_count[id_rosto] = id_count.get(id_rosto, 0) + 1
                print(f"➡ Foto salva: {foto_filename} | Aparições ID {id_rosto}: {id_count[id_rosto]}")
                log_rosto(video_filename, id_rosto, foto_filename)

            gravando = False
            ultima_gravacao = time.time()
            video_count += 1

    status_text = "Detecção: ON" if detecção_ativa else "Detecção: OFF"
    cv2.putText(frame, status_text, (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Detecção Facial com Interface Estável", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        detecção_ativa = True
        print("✅ Detecção iniciada")
    elif key == ord('p'):
        detecção_ativa = False
        print("⏸️ Detecção pausada")

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
