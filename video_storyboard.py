from datetime import datetime, timedelta
from pathlib import Path

import cv2


def video_storyboard(videos_folder: str, output_folder: str, save_every_n_shot: int, fps: int = 30, img_size: list = None):
    """Функция проходит циклом по всем файлам указанной папки\n
    Проверяет формат названий видео: он должен быть 2026_01_20_14-45 такого вида, где 14-45 это время начала записи видео\n
    Далее происходит раскадровка, где имя картинки создается путем расчета точного времени кадра\n
    Аргументы:\n
            output_folder (str): папка, в которую будет все сохраняться\n
            save_every_n_shot (int): какой кадр по очереди сохранять (Например каждый третий)\n
            fps (int): fps видео файлов\n
            img_size [int, int]: При необходимости можно изменить размер линейным алгоритмом\n
    Видео-файлы могут быть любого формата, поддерживаемого OpenCV 
    """

    for video_path in Path(videos_folder).iterdir():

        try:
            start_time = datetime.strptime(video_path.stem, '%Y_%m_%d_%H-%M')
        except ValueError:
            print(f'Пропуск файла {video_path.name}: неверный формат даты в названии.')
            continue

        current_output_folder = Path(output_folder) / video_path.stem
        current_output_folder.mkdir(parents=True, exist_ok=True)

        video = cv2.VideoCapture(video_path)

        real_shot_counter = 0
        saved_count = 0

        while True:
            bool_, shot = video.read()
            if not bool_:
                break

            if real_shot_counter % save_every_n_shot == 0:
                seconds_passed = real_shot_counter / fps
                current_shot_time = start_time + timedelta(seconds=seconds_passed)
                time_str = current_shot_time.strftime('%Y-%m-%d__%H-%M-%S-%f')[:-3]

                if img_size:
                    shot = cv2.resize(shot, img_size)

                save_name = current_output_folder / f'{time_str}.jpg'
                cv2.imwrite(save_name, shot)
                saved_count += 1

            real_shot_counter += 1

        video.release()
        print(f'Файл {video_path.name} раскадрирован. Создано {saved_count} изображений из {real_shot_counter} кадров.')
