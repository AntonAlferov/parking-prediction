config = {
    'format_video_out': 'mp4v',                                                 # Формат выходного видео
    'video_path': 'Full_Video/2026_04_02/2026_04_02_14-27.mp4',                 # Путь до исходного видео
    'output_path': 'sait.mp4',                                                  # Имя видео поcле обработки
    'device': 'cuda:0',                                                         # Устройство для обработки
    'db_car_file': 'car_base.csv',                                              # Путь до файла базы автомобилей
    'db_parking_file': 'parking_base.csv',                                      # Путь до файла базы парковок

    'confidence_threshold_yolo': 0.7,      # Порог вероятностей для YOLO
    'threshold_confidence_zone': 0.8,      # Порог определения парковочных зон
    'threshold_reid': 0.6,                 # Порог схожести реидентификации машин в зоне
    'threshold_reid_base': 0.6,            # Порог схожести реидентификации машин при записи в базу
    'confidence_license_plate': 0.6,       # Порог уверенности в распознавании автомобильного номера

    'min_box_car_area': 500,               # Минимальная площадь боксов автомобилей
    'transform_box': train_transforms,     # Алгоритм трансформации боксов для модели ReiD
    'batch_size': 32,                      # Количество батчей обработки боксов
    'bottom_line': 200,                    # Отступ от низа кадра с которой детектируются автомобили
    'max_tolerance_zone': 30,              # Терпимость (в кадрах) к помехам модели детекции парковочнх зон

    'width_text_space_over_box': 460,      # Ширина плашек с текстом над боксами
    'height_text_space_over_box': 80,      # Высота плашек с текстом над боксами
    'font_scale': 1,                       # Масштаб шрифта
    'thickness': 1,                        # Толщина шрифта
    'font': cv2.FONT_HERSHEY_COMPLEX,      # Шрифт
    'panel_width': 700,                    # Ширина информационной панели
    'panel_height': 350,                   # Высота информационной панели
    'panel_pad': 20                        # Отступ панели от края
}