import os

import cv2
from PIL import Image
from tqdm import tqdm


def recording_boundig_boxes(model, video_path, out_box_img_path, min_area_box=4000, confidence_threshold=0.65, max_box_count=1000000, target_class='car', shot_step=3):
    """
    Данная функция делает раскадровку видео,\n
    В каждом кадре предказываются баунднг боксы\n
    Боксы нужного класса сохраняются в указанную папку, имя изобраений задается последовательной нумерацией\n
    Аргументы:\n
            model (model obj): модель.\n
            video_path (str): Путь до файла видео.\n
            out_box_img_path (str): Название папки для сохранения кадров.\n
            min_area_box (int): Минимальная площадь боксов\n
            confidence_threshold (int): Порог вероятности бокса\n
            max_box_count (int): Максимальное колличество созданных картинок\n
            target_class (str): Имя целевого класса\n
            shot_step (int): Шаг обработки кадров\n
    """

    out_path = f'{out_box_img_path}/{target_class}'
    os.makedirs(out_path, exist_ok=True)

    box_count = 0
    shot_count = 0

    pbar = tqdm(range(max_box_count))

    cap = cv2.VideoCapture(video_path)

    while True:
        bool_, shot = cap.read()
        if not bool_ or box_count >= max_box_count:
            break

        shot_count += 1

        if shot_count % shot_step == 0:

            shot_rgb = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
            pil_shot = Image.fromarray(shot_rgb)
            result = model.predict(source=pil_shot, conf=confidence_threshold, verbose=False)

            for box in result[0].boxes:

                if box_count >= max_box_count:
                    break

                class_id = int(box.cls[0])
                detect_name = model.names[class_id]

                if detect_name == target_class:

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    if area >= min_area_box:

                        save_path = f'{out_path}/{box_count}.jpg'
                        cut_box = shot[y1:y2, x1:x2]

                        cv2.imwrite(save_path, cut_box)

                        box_count += 1
                        pbar.update(1)

    cap.release()
    pbar.close()