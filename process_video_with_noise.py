import os

import cv2
import numpy as np
from tqdm import tqdm


def process_video_with_noise(model, video_path, output_folder, conf_threshold=0.4, box_padding=15, noise_sigma=30, shot_step=10):
    """
    Данная функция делает раскадровку видео,\n
    обрабатывает кадры указанной моделью, заполняя шумом(средний цвет кадра) вычисленные баундинг боксы\n
    Обработанные кадры задаются именем номера кадра и сохраняются\n
    Аргументы:\n
            model (model obj): модель.\n
            video_path (str): Путь до файла видео.\n
            output_folder (str): Название папки для сохранения кадров.\n
            conf_threshold (float): Порог вероятности (уверенности) модели\n
            box_padding (int): Отступ от баундинг бокса(Чтобы полностью перекрыть авто)\n
            noise_sigma (float): Зернистость шума\n
            shot_step (int): Шаг раскадровки\n
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)

    total_shots = video.get(cv2.CAP_PROP_FRAME_COUNT)
    shot_index = 0

    pbar = tqdm(total=total_shots, unit='кадров', desc='Прогресс', file=sys.stdout, mininterval=1)

    while True:
        bool_, shot = video.read()
        if not bool_:
            break

        if shot_index % shot_step == 0:
            h_img, w_img, _ = shot.shape

            mean_color = np.array(cv2.mean(shot)[:3])

            results = model.predict(shot, conf=conf_threshold, verbose=False)

            for result in results:

                boxes = result.boxes
                for box in boxes:

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    x1 = max(0, x1 - box_padding)
                    y1 = max(0, y1 - box_padding)
                    x2 = min(w_img, x2 + box_padding)
                    y2 = min(h_img, y2 + box_padding)

                    bw = x2 - x1
                    bh = y2 - y1

                    noise = np.random.normal(loc=mean_color, scale=noise_sigma, size=(bh, bw, 3))

                    shot[y1:y2, x1:x2] = noise

            filename = f'{output_folder}/clean_{shot_index}.jpg'
            cv2.imwrite(filename, shot)
        shot_index += 1
        pbar.update(1)

    pbar.close()
    video.release()
    print(f'Раскадровка выполнена. Баундинг боксы заменены на шум обработано {shot_index} кадров')
