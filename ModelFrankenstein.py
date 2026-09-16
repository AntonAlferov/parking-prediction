from datetime import datetime, timedelta
from pathlib import Path


import cv2
import pandas as pd
import torch
from PIL import Image
from torch.nn.functional import normalize
from tqdm import tqdm


class ModelFrankenstein():
    def __init__(self, model_box_detected, model_detect_zone, model_reid_car):
        """
            Инициализирует основной класс-обработчик, принимая модели для детекции,\n
            определения зон и ReID. Подготавливает структуры данных (словари и списки)\n
            для накопления статистики, карт отрисовки и временных данных.
        """

        self.config = config

        self.model_box_detected = model_box_detected
        self.model_detect_zone = model_detect_zone
        self.model_reid_car = model_reid_car

        self.db_car = None
        self.db_parking = None
        self.checking_cars = []
 
        self._shot_id = -1
        self._cumulative_shot_base = {}
        self._cumulative_car_base = {}
        self._temp_license_plates = {}
        self._drawing_box_map = {}
        self._drawing_panel_info_map = {}
        self._update_id = {}

        self.video_handler()

    def video_handler(self):
        """
            Настраивает параметры захвата (input) и записи (output) видеопотока.\n
            Считывает метаданные видео (разрешение, FPS), вычисляет границы зон отрисовки\n
            и инициализирует генератор трекинга объектов (YOLO) для покадровой обработки.
        """
        cap = cv2.VideoCapture(self.config['video_path'])
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.shot_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.bottom_line = self.video_height - self.config['bottom_line']
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*self.config['format_video_out'])
        self.out_video = cv2.VideoWriter(self.config['output_path'], fourcc, self.fps, (self.video_width, self.video_height))
        self.results_generator = self.model_box_detected.track(source=self.config['video_path'], imgsz=1280, conf=self.config['confidence_threshold_yolo'], stream=True, verbose=False, persist=True)

    def predict_zone(self, shot):
        """
            Определяет принадлежность текущего кадра к конкретной зоне парковки.\n
            Преобразует формат изображения (BGR -> RGB -> PIL) и выполняет инференс\n
            модели классификации зон, возвращая название улицы и уверенность предсказания.
        """

        self.model_detect_zone.to(self.config['device']).eval()

        shot_rgb = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(shot_rgb)

        street_name, score = predict_parking_zone(pil_image, self.model_detect_zone)

        return street_name, score

    def cumulative_base(self, boxes):
        """
            Фильтрует и распределяет сырые детекции объектов (автомобили и номера).\n
            Исключает объекты, касающиеся краев кадра, и автомобили с недостаточной площадью.\n
            Валидные детекции разделяются по классам и сохраняются во временные хранилища\n
            (`cumulative_car_base` для авто, `temp_license_plates` для номеров) с привязкой к ID кадра.
        """

        margin = 5

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0])

            width = x2 - x1
            height = y2 - y1
            area = width * height

            is_touching_edge = (
                x1 < margin or
                y1 < margin or
                x2 > self.video_width - margin or
                y2 > self.video_height - margin
            )

            if is_touching_edge:
                continue

            if class_id == 0:
                if area >= self.config['min_box_car_area']:
                    if y2 > self.bottom_line:            
                        if not self._shot_id in self._cumulative_car_base:
                            self._cumulative_car_base[self._shot_id] = []
                        self._cumulative_car_base[self._shot_id].append([(x1, y1, x2, y2), area, confidence])
            elif class_id == 1:
                if not self._shot_id in self._temp_license_plates:
                    self._temp_license_plates[self._shot_id] = []
                self._temp_license_plates[self._shot_id].append(((x1, y1, x2, y2), confidence))

    def binding_license_plates_to_car(self):
        """
            Связывает детектированные номера с соответствующими автомобилями.
            Использует геометрическую проверку: номер считается принадлежащим автомобилю, 
            если центр bounding box номера находится внутри bounding box автомобиля. 
            Дополняет данные об авто найденным номером (или `None`) и очищает буфер номеров.
        """

        for index, car_data in self._cumulative_car_base.items():

            for car_info in car_data:

                car_x1, car_y1, car_x2, car_y2 = car_info[0]

                if index in self._temp_license_plates:
                    for plate_info in self._temp_license_plates[index]:

                        p_x1, p_y1, p_x2, p_y2 = plate_info[0]

                        p_center_x = (p_x1 + p_x2) / 2
                        p_center_y = (p_y1 + p_y2) / 2

                        if car_x1 < p_center_x < car_x2 and car_y1 < p_center_y < car_y2:

                            car_info.append(plate_info)
                            break

                if len(car_info) < 4:
                    car_info.append(None)

        self._temp_license_plates.clear()

    def reid_car(self):
        """
            Выполняет идентификацию (ReID) автомобилей и присваивает им уникальные классы.\n
            Формирует пакеты (батчи) из вырезанных изображений авто.\n
            Генерирует векторные признаки (эмбеддинги). Затем группирует\n
            похожие объекты: если новый вектор совпадает с существующей группой, её центр\n
            обновляется методом скользящего среднего. Если совпадений нет — создается новая\n
            группа (новый уникальный класс).
         """

        self.model_reid_car.to(self.config['device']).eval()

        tensor_list = []
        car_references = []

        for shot_id, car_list in self._cumulative_car_base.items():
            for car_info in car_list:
                x1, y1, x2, y2 = car_info[0]

                clipping = self._cumulative_shot_base[shot_id][0][y1:y2, x1:x2]

                result = self.config['transform_box'](image=clipping)
                tensor_list.append(result['image'])

                car_references.append(car_info)

        full_cpu_tensor = torch.stack(tensor_list) 
        embedding_results = []

        with torch.no_grad():
            for i in range(0, len(full_cpu_tensor), self.config['batch_size']):
                batch = full_cpu_tensor[i : i + self.config['batch_size']].to(self.config['device'])
                emb_batch = self.model_reid_car(batch)
                embedding_results.append(emb_batch) 

        embeddings = torch.cat(embedding_results)
        embeddings = normalize(embeddings, p=2, dim=1)

        cluster_centers = []
        cluster_counts = []

        for index, embedding in enumerate(embeddings):

            current_car_info = car_references[index]

            best_id = -1
            if cluster_centers:
                similarities = torch.mv(torch.stack(cluster_centers), embedding)
                best_score, best_index = torch.max(similarities, dim=0)

                if best_score > self.config['threshold_reid']:
                    best_id = best_index.item()

            if best_id != -1:
                cnt = cluster_counts[best_id]
                cluster_centers[best_id] = (cluster_centers[best_id] * cnt + embedding) / (cnt + 1)
                cluster_centers[best_id] = normalize(cluster_centers[best_id], p=2, dim=0)
                cluster_counts[best_id] += 1
            else:
                cluster_centers.append(embedding)
                cluster_counts.append(1)
                best_id = len(cluster_centers) - 1

            final_embedding = cluster_centers[best_id].detach().cpu().tolist()

            if len(current_car_info) > 4:
                current_car_info[-1] = best_id
                if len(current_car_info) > 5:
                    current_car_info[5] = final_embedding
                else:
                    current_car_info.append(final_embedding)
            else:
                current_car_info.append(best_id)
                current_car_info.append(final_embedding)

    def _load_or_create_base(self, file_name, columns, time=False):
        """
            Служебный метод для безопасной загрузки данных.(Вспомогательная)\n
            Проверяет наличие CSV-файла по указанному пути. Если файл существует, загружает его\n
            в pandas DataFrame (с опциональным парсингом дат). Если файла нет, создает пустой\n 
            DataFrame с заданной структурой колонок.
        """

        if os.path.exists(file_name):
            if time:
                return pd.read_csv(file_name, parse_dates=['time'])
            return pd.read_csv(file_name)
        return pd.DataFrame(columns=columns)

    def load_or_create_base(self):
        """
            Инициализирует рабочие базы данных проекта.(Основная)\n
            Определяет схему таблиц (список колонок) для базы автомобилей и журнала парковок.\n
            Вызывает служебный метод для загрузки существующих данных с диска или создания\n 
            новых таблиц, сохраняя их в атрибуты класса.
        """

        col_car_base = ['id', 'embeding', 'license_plate', 'confidence_plate', 'image']
        col_parking_base = ['time', 'parking_name', 'id_car', 'confidence_car']

        self.db_car = self._load_or_create_base(self.config['db_car_file'], col_car_base)
        self.db_parking = self._load_or_create_base(self.config['db_parking_file'], col_parking_base, time=True)

    def _get_unique_id(self):
        """
            Генерирует уникальный 4-значный строковый идентификатор.\n
            Создает случайную комбинацию из букв и цифр. Гарантирует уникальность ID,\n 
            проверяя его отсутствие в текущей базе данных автомобилей перед возвратом.
        """

        while True:
            new_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            if self.db_car.empty or new_id not in self.db_car['id'].values:
                return new_id

    def license_plate_recognition(self):
        """
            Выбирает лучший кадр с номером для каждого уникального автомобиля и распознает текст.\n
            Анализирует все накопленные детекции и находит для каждого кластера (автомобиля)\n 
            изображение номера с наивысшей уверенностью модели. Выполняет оптическое распознавание\n 
            символов (PaddleOCR) только на этих лучших образцах, после чего присваивает распознанный\n 
            номер всем записям данного автомобиля в базе.
        """

        recognizer = License_Plate_Recognition(self.config['confidence_license_plate'])

        max_plate_confidence = {}

        for shot_id, car_list in self._cumulative_car_base.items():
            for car_info in car_list:

                if car_info[3]:
                    current_cluster_index = car_info[4]
                    current_confidence_plate = car_info[3][1]
                    current_box_plate = car_info[3][0]

                    if (
                        current_cluster_index in max_plate_confidence and
                        current_confidence_plate > max_plate_confidence[current_cluster_index][0]
                        ):
                            max_plate_confidence[current_cluster_index] = [current_confidence_plate, shot_id, current_box_plate]
                    elif current_cluster_index not in max_plate_confidence:
                        max_plate_confidence[current_cluster_index] = [current_confidence_plate, shot_id, current_box_plate]

        for cluster, plate_info in max_plate_confidence.items():

            x1, y1, x2, y2 = plate_info[2]
            shot_id = plate_info[1]
            confidence_plate = plate_info[0]
            clipping_plate = self._cumulative_shot_base[shot_id][0][y1:y2, x1:x2]

            found_plate = recognizer.recognize(clipping_plate)

            max_plate_confidence[cluster] = [found_plate, confidence_plate]

        for shot_id, car_list in self._cumulative_car_base.items():

            for car_info in car_list:

                cluster = car_info[4]
                if car_info[3]:
                    if cluster in max_plate_confidence:
                        car_info[3] = max_plate_confidence[cluster] 

    def get_last_parking_info(self, id_car, current_parking_name):
        """
        Находит время начала НЕПРЕРЫВНОЙ парковочной сессии для машины.\n
        Проверяет, что последняя запись о машине соответствует текущей парковке,\n
        затем ищет самую раннюю запись в этой непрерывной сессии.\n
        возвращает Объект datetime с временем заезда или None, если машина не найдена\n
        на этой парковке в своей последней сессии
        """
        if self.db_parking.empty:
            return None

        car_records = self.db_parking[self.db_parking['id_car'] == id_car]

        if car_records.empty:
            return None

        last_record = car_records.iloc[-1]
        if last_record['parking_name'] != current_parking_name:
            return None

        beginning_time = None
        for _, row in car_records.iloc[::-1].iterrows():
            if row['parking_name'] == current_parking_name:
                beginning_time = row['time']
            else:
                break

        return beginning_time

    def synchronization_accumulated_and_pd_bases(self):
        """
            Синхронизирует обнаруженные автомобили с постоянной базой данных и фиксирует события.\n

            Сравнивает эмбеддинги текущих объектов с архивом (Pandas DataFrame) по косинусному сходству.\n

            Логика работы:
            1. **Поиск совпадений:** Если автомобиль найден, подтягивает его ID и проверяет соответствие госномера.\n
            При конфликте номеров сохраняет отладочные изображения для ручной проверки. Рассчитывает время с последнего визита.\n
            2. **Регистрация новых:** Если совпадений нет, генерирует новый ID, сохраняет лучшее изображение автомобиля на диск и добавляет запись в реестр.\n
            3. **Журналирование:** Добавляет событие (время, место, ID) в таблицу парковок.\n
            4. **Сохранение:** Перезаписывает обновленные базы данных в CSV-файлы.
         """

        car_set = set()

        for shot_id, car_list in self._cumulative_car_base.items():
            for car_info in car_list:
                cluster_index = car_info[4]
                if cluster_index in car_set:
                   continue
                car_set.add(cluster_index)

                current_emb = np.array(car_info[5])

                current_license_plate = None
                if car_info[3] and car_info[3][0] is not None :
                    current_license_plate = car_info[3][0]


                found_id = None
                max_score = -1
                plate_in_base = None
                license_plate = None
                confidence_plate = None

                matching_license_plate = None

                if car_info[3]:
                    license_plate = car_info[3][0] 
                    confidence_plate = car_info[3][1]

                if not self.db_car.empty:
                    for col, row in self.db_car.iterrows():

                        val_emb = row['embedding']
                        db_emb = eval(val_emb) if isinstance(val_emb, str) else val_emb

                        norm = np.linalg.norm(current_emb) * np.linalg.norm(db_emb)                   
                        score = np.dot(current_emb, db_emb) / norm

                        if score > self.config['threshold_reid_base'] and score > max_score:
                            max_score = score
                            found_id = row['id']
                            if row['license_plate'] is None and current_license_plate is None:
                                matching_license_plate = 'update WITHOUT PLATE'
                                plate_in_base = row['license_plate']
                            elif row['license_plate'] == current_license_plate:
                                matching_license_plate = 'matching OK'
                                plate_in_base = row['license_plate']
                            elif row['license_plate'] is None and current_license_plate is not None:
                                matching_license_plate = 'update CHECK'
                                plate_in_base = row['license_plate']

                                save_dir_check = f'check_car/none_vs_not_none/{found_id}'
                                os.makedirs(save_dir_check, exist_ok=True)
                                current_img = self.get_best_img_car()[cluster_index]
                                path_in_base = row['image']
                                img_in_base = cv2.imread(path_in_base)
                                cv2.imwrite(f'{save_dir_check}/new_car.jpg', current_img)
                                cv2.imwrite(f'{save_dir_check}/car_in_base.jpg', img_in_base)

                            elif row['license_plate'] and  current_license_plate is None:
                                matching_license_plate = 'update  OK'
                                plate_in_base = row['license_plate']
                            elif row['license_plate'] and current_license_plate and row['license_plate'] != current_license_plate:
                                matching_license_plate = 'update CHECK'
                                plate_in_base = row['license_plate']

                                save_dir_check = f'check_car/plate_not_equal/{found_id}'
                                os.makedirs(save_dir_check, exist_ok=True)
                                current_img = self.get_best_img_car()[cluster_index]
                                path_in_base = row['image']
                                img_in_base = cv2.imread(path_in_base)
                                cv2.imwrite(f'{save_dir_check}/new_car.jpg', current_img)
                                cv2.imwrite(f'{save_dir_check}/car_in_base.jpg', img_in_base)

                if found_id:
                    self._update_id[cluster_index] = found_id, plate_in_base, matching_license_plate
                    final_id = found_id
                    last_parking_date = self.get_last_parking_info(final_id, self._cumulative_shot_base[shot_id][1])
                    if last_parking_date:
                        time_difference = self._cumulative_shot_base[shot_id][2] - last_parking_date
                        total_diff_second = int(time_difference.total_seconds())
                        hours = total_diff_second // 3600
                        minutes = (total_diff_second % 3600) // 60
                        fin_diff = f'{hours}ч {minutes}мин'
                    else:
                        fin_diff = '0ч 0мин'

                    self._update_id[cluster_index] = found_id, plate_in_base, matching_license_plate, fin_diff
                else:
                    final_id = self._get_unique_id()
                    best_imgs = self.get_best_img_car()[cluster_index]

                    save_dir = 'saved_car'
                    os.makedirs(save_dir, exist_ok=True)
                    img_path = f'{save_dir}/car_id_{final_id}.jpg'
                    cv2.imwrite(img_path, best_imgs)
                    self._update_id[cluster_index] = final_id, plate_in_base, matching_license_plate, '0ч 0мин'
                    new_row_car = pd.DataFrame([{
                        'id': final_id,
                        'embedding': current_emb.tolist(),
                        'license_plate': license_plate,
                        'confidence_plate': confidence_plate,
                        'image': img_path
                    }])
                    if self.db_car.empty:
                        self.db_car = new_row_car
                    else:
                        self.db_car = pd.concat([self.db_car, new_row_car], ignore_index=True)

                new_row_parking = pd.DataFrame([{
                        'time': self._cumulative_shot_base[shot_id][2],
                        'parking_name': self._cumulative_shot_base[shot_id][1],
                        'id_car': final_id,
                        'confidence_car': car_info[2],
                    }])
                if self.db_parking.empty:
                    self.db_parking = new_row_parking
                else:
                    self.db_parking = pd.concat([self.db_parking, new_row_parking], ignore_index=True)

        self.db_parking.to_csv(self.config['db_parking_file'], index=False)
        self.db_car.to_csv(self.config['db_car_file'], index=False)

    def get_best_img_car(self):
        """
            Извлекает эталонное изображение для каждого уникального автомобиля.\n

            Анализирует все накопленные кадры и выбирает для каждого кластера (автомобиля)\n 
            детекцию с наивысшим коэффициентом уверенности. Вырезает изображение объекта,\n 
            масштабирует его до фиксированной высоты (180px) с сохранением пропорций и\n 
            возвращает словарь лучших снимков.
        """

        best_car_info = {}

        for shot_id in self._cumulative_shot_base.keys():

            if shot_id in self._cumulative_car_base:

                for items in self._cumulative_car_base[shot_id]:

                    box = items[0]
                    confidence = items[2]
                    cluster_id = items[4]

                    if confidence > best_car_info.get(cluster_id, (0,))[0]:
                        best_car_info[cluster_id] = confidence, box, shot_id

        best_car_img = {}

        for cluster_id, car_info in best_car_info.items():

            x1, y1, x2, y2 = car_info[1]
            clipping_car = self._cumulative_shot_base[car_info[2]][0][y1:y2, x1:x2]

            target_height = 180
            h, w = clipping_car.shape[:2]

            if h > target_height:
                ratio = target_height / h
                new_with = int(w * ratio)

                resize_img = cv2.resize(clipping_car, (new_with, target_height), interpolation=cv2.INTER_AREA)
            else:
                resize_img = clipping_car

            best_car_img[cluster_id] = resize_img

        return best_car_img

    def data_for_drow(self):
        """
            Агрегирует данные для финальной визуализации на видео.\n

            Сопоставляет накопленные детекции с обновленной информацией из базы данных\n 
            (присвоенные ID, статусы сверки номеров, время стоянки). Формирует структуру\n 
            `_drawing_box_map`, содержащую координаты рамок, цвета индикаторов и текстовые\n 
            подписи для каждого кадра, подготавливая их к отрисовке.
        """

        for shot_index, car_info in self._cumulative_car_base.items():

            for box, _, _, plate, cluster_index, _ in car_info:

                color = 255, 140, 0
                current_plate = 'Не распознан'
                if plate and plate[0] is not None:
                    current_plate = plate[0]
                    color = 0, 255, 0

                if shot_index not in self._drawing_box_map:
                    self._drawing_box_map[shot_index] = []

                update_info = self._update_id[cluster_index]
                new_id = update_info[0]
                status_matching_license_plate = update_info[2]
                date_time = update_info[3]

                self._drawing_box_map[shot_index].append((box, color, new_id, date_time, 'None', status_matching_license_plate, current_plate))

    def draw_box(self):
        """
            Наносит визуальную разметку на кадры видеопотока.\n

            Итерируется по списку кадров и рисует цветные ограничивающие рамки (bounding boxes)\n 
            вокруг автомобилей. Поверх рамок создает информационные плашки с черным фоном,\n 
            отображающие ID, время стоянки и прогноз выезда. Включает логику коррекции координат,\n 
            чтобы текст не выходил за пределы экрана.
        """

        for shot_index, shot_info in self._cumulative_shot_base.items():

            shot_to_write = shot_info[0]

            if shot_index in self._drawing_box_map:

                draw_boxes_info = self._drawing_box_map[shot_index]

                for box_info in draw_boxes_info:
                    box, color, cluster_index, text_time, text_pred, _, _ = box_info
                    x1, y1, x2, y2 = box
                    cv2.rectangle(shot_to_write, (x1, y1), (x2, y2), color, 2)

                    space_x1 = x1
                    space_y1 = y1 - self.config['height_text_space_over_box'] - 10
                    space_x2 = space_x1 + self.config['width_text_space_over_box']
                    space_y2 = space_y1 + self.config['height_text_space_over_box']

                    if space_y1 < 0:
                        shift = -space_y1
                        space_y1 += shift
                        space_y2 += shift

                    if space_x2 > self.video_width:
                        shift = space_x2 - self.video_width
                        space_x1 -= shift
                        space_y2 -= shift

                    cv2.rectangle(shot_to_write,
                                (space_x1, space_y1),
                                (space_x2, space_y2),
                                (0, 0, 0), cv2.FILLED)

                    text_x = space_x1 + 10
                    base_y = space_y1 + 25 

                    line1 = f"ID: {cluster_index}"
                    line2 = f"Время стоянки: {text_time}"
                    line3 = f"Прогноз выезда: {text_pred}"

                    cv2.putText(shot_to_write, line1, (text_x, base_y),      self.config['font'], self.config['font_scale'], (255, 255, 255), self.config['thickness'], cv2.LINE_AA)
                    cv2.putText(shot_to_write, line2, (text_x, base_y + 25), self.config['font'], self.config['font_scale'], (255, 255, 255), self.config['thickness'], cv2.LINE_AA)
                    cv2.putText(shot_to_write, line3, (text_x, base_y + 50), self.config['font'], self.config['font_scale'], (255, 255, 255), self.config['thickness'], cv2.LINE_AA)

    def draw_panel_info(self, current_shot=None):
        """
            Отрисовывает информационную панель.\n

            Создает полупрозрачную боковую панель со статистикой по текущей зоне парковки\n 
            (название зоны, счетчик машин).\n

            Реализует логику "триггерной линии":\n
            1. Отслеживает автомобили, пересекшие условную линию в нижней части кадра.\n
            2. Выбирает наиболее актуальный объект (ближайший к низу).\n
            3. Выводит расширенную информацию о нем на панель: миниатюру (кроп), ID, госномер,\n 
            время фиксации и статус совпадения с базой.\n

            Если зона не распознана (`current_shot` передан), выводит предупреждение об отсутствии зоны в базе.\n
            Записывает итоговый кадр в выходной видеофайл.
        """
        pad = self.config['panel_pad']
        alpha = 0.6

        panel_x1 = self.video_width - self.config['panel_width'] - pad
        panel_y1 = pad
        panel_x2 = self.video_width - pad
        panel_y2 = pad + self.config['panel_height']

        threshold_line = self.video_height - 160

        car_id_set = set()

        if current_shot is None:
            for shot_id, shot_info in self._cumulative_shot_base.items():
                shot_to_write, street_name, _ = shot_info

                overlay = shot_to_write.copy()
                cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
                shot_to_write = cv2.addWeighted(overlay, alpha, shot_to_write, 1 - alpha, 0)

                street_x_start = panel_x1 + 20
                cv2.putText(shot_to_write, f'Зона парковки: {street_name}', (street_x_start, panel_y1 + 40), self.config['font'], 1.0, (0, 255, 0), 2)
                cv2.putText(shot_to_write, f'Найдено машин: {len(car_id_set)}', (street_x_start, panel_y1 + 80), self.config['font'], 1.0, (0, 255, 0), 2)

                if shot_id in self._drawing_box_map:
                    cars_info_in_shot = self._drawing_box_map[shot_id]

                    cars_below_line = [car_info for car_info in cars_info_in_shot if car_info[0][3] > threshold_line]                

                    if cars_below_line:
                        target_car_info = sorted(cars_below_line, key=lambda car: car[0][3], reverse=True)[0]

                        car_id = target_car_info[2]
                        car_id_set.add(car_id)

                        parking_record = self.db_parking[self.db_parking['id_car'] == car_id]

                        car_date = parking_record['time'].iloc[-1]
                        car_record_df = self.db_car[self.db_car['id'] == car_id]

                        car_img_path = car_record_df['image'].iloc[-1]
                        car_img = cv2.imread(car_img_path)

                        h, w, _ = car_img.shape

                        current_y_base = panel_y1 + 120

                        y_img = int(current_y_base)
                        x_img = int(panel_x1 + 50)

                        if y_img + h < shot_to_write.shape[0] and x_img + w < shot_to_write.shape[1]:
                            shot_to_write[y_img:y_img + h, x_img:x_img + w] = car_img

                        text_offset_x = w + 20
                        text_y = current_y_base + 20
                        line_height = 30

                        date_str = str(car_date).split('.')[0]
                        cv2.putText(shot_to_write, f'ID: {car_id}', (x_img + text_offset_x, text_y), self.config['font'], 0.6, (255, 255, 255), 1)               
                        cv2.putText(shot_to_write, f'Гос. Номер: {target_car_info[6]}', (x_img + text_offset_x, text_y + line_height), self.config['font'], 0.6,  target_car_info[1], 1)                     
                        cv2.putText(shot_to_write, f'Время: {date_str}', (x_img + text_offset_x, text_y + line_height * 2), self.config['font'], 0.5, (200, 200, 200), 1)
                        cv2.putText(shot_to_write, f'Совпадение в базе: {target_car_info[5]}', (x_img + text_offset_x, current_y_base + 110), self.config['font'], 0.5, (255, 255, 255), 1)

                self.out_video.write(shot_to_write)
        else:            
            overlay = current_shot.copy()
            cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
            current_shot = cv2.addWeighted(overlay, alpha, current_shot, 1 - alpha, 0)

            street_x_start = panel_x1 + 20
            cv2.putText(current_shot, 'Данной зоны нет в базе', (street_x_start, panel_y1 + 40), self.config['font'], 1.0, (0, 0, 255), 2)

            self.out_video.write(current_shot)

    def update_zone_state(self, score, current_toleransce, current_in_zone):
        """
            Обновляет статус нахождения в зоне, сглаживая кратковременные сбои.\n

            Использует счетчик стабильности: если уверенность детекции высокая — счетчик уменьшается,\n 
            если низкая — растет. Переключает статус «в зоне / не в зоне» только тогда,\n 
            когда счетчик достигает предела. Это предотвращает случайные переключения\n 
            из-за одного неудачного кадра.
        """

        if score > self.config['threshold_confidence_zone']:
            if current_toleransce > 0:
                current_toleransce -= 1
        else:
            if current_toleransce != self.config['max_tolerance_zone']:
                current_toleransce += 1

        if current_toleransce == 0:
            current_in_zone = True
        elif current_toleransce == self.config['max_tolerance_zone']:
            current_in_zone = False

        return current_toleransce, current_in_zone

    def logic_processing_accumulated_data(self):
        """
            Запускает конвейер обработки накопленных данных.\n

            Последовательно вызывает методы для:\n
            1. Привязки госномеров к автомобилям.\n
            2. ReID (повторной идентификации) и распознавания номеров.\n
            3. Синхронизации с базой данных (регистрация/обновление).\n
            4. Подготовки и отрисовки визуализации.\n

            В конце очищает буферы накопленных данных (`_cumulative_car_base`, `_cumulative_shot_base`)\n
            для приема следующей пачки кадров.
        """

        self.binding_license_plates_to_car()
        self.reid_car()
        self.license_plate_recognition()
        self.synchronization_accumulated_and_pd_bases()
        self.data_for_drow()
        self.draw_box()
        self.draw_panel_info()

        self._cumulative_car_base.clear()
        self._cumulative_shot_base.clear()

    def calculating_shot_date(self, shot_index):
        """
            Вычисляет реальное время для конкретного кадра.\n

            Парсит дату и время начала записи из названия видеофайла (формат: %Y_%m_%d_%H-%M).\n
            Рассчитывает смещение времени на основе индекса кадра и FPS видеопотока,\n 
            возвращая точный `datetime` объект для текущего момента.
        """

        try:
            start_time = datetime.strptime(Path(self.config['video_path']).stem, '%Y_%m_%d_%H-%M')
        except ValueError:
            print('Ошибка неверный формат даты в названии обрабатываемого файла.')

        seconds_passed = shot_index / self.fps
        current_shot_time = start_time + timedelta(seconds=seconds_passed)

        return current_shot_time

    def predict(self):
        """Запускает основной цикл обработки кадров видеопотока\n
        Метод итерируется по результатам генерации детекции, определяет текущую зону парковки\n
        и управляет логикой накопления данных. Если кадр находится в зоне интереса,\n
        происходит сбор статистики. При выходе из зоны или завершения потокаинициализируется\n
        обработка накопленных данных и отрисовка информации
        """

        self.load_or_create_base()

        pbar = tqdm(range(self.shot_count))

        max_tolerance_zone = self.config['max_tolerance_zone']
        in_zone = False

        for result in self.results_generator:

            self._shot_id += 1
            pbar.update(1)

            street_name, score = self.predict_zone(result.orig_img)           
            max_tolerance_zone, in_zone = self.update_zone_state(score, max_tolerance_zone, in_zone)             

            if in_zone:
                current_shot_date = self.calculating_shot_date(self._shot_id)
                self._cumulative_shot_base[self._shot_id] = [result.orig_img.copy(), street_name, current_shot_date]
                self.cumulative_base(result.boxes)

            else:
                if self._cumulative_car_base:
                    self.logic_processing_accumulated_data()
                else:
                    self.data_for_drow()
                    self.draw_panel_info(result.orig_img)

        if self._cumulative_car_base:
            self.logic_processing_accumulated_data()

        pbar.close()
        self.out_video.release()