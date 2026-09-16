import re

import cv2
from paddleocr import PaddleOCR


class License_Plate_Recognition():
    """Класс для распознавания автомобильных номеров с помощью PaddleOCR"""

    def __init__(self, confidence_license_plate):
        """
        Инициализирует модуль OCR с классификацией угла поворота и загружает веса моделей
        """

        self.ocr = PaddleOCR(
                text_detection_model_dir='OCRv5\PP-OCRv5_server_det_infer',                             # Модель детектирования
                text_recognition_model_dir='OCRv5\PP-OCRv5_server_rec',                                 # Модель распознавания
                textline_orientation_model_dir='OCRv5\PP-LCNet_x1_0_textline_ori',                      # Модуль ориентации текста
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                return_word_box=False,
                text_det_limit_side_len=64,
                text_rec_score_thresh=confidence_license_plate                                          # Порог вероятности распознавания текста
                )

        self.license_pattern = re.compile(r"^[А-Я]{1}\d{3}[А-Я]{2}\d{2,3}$", re.IGNORECASE)

    def preprocess_for_ocr(self, image, target_height=64):
        """
        Предобработка вырезанного номера для OCR:\n
        изменение размера, удаление шума, повышение контрастности\n
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        if not height:
            return None
        scale_ratio = target_height / height
        new_width = int(width * scale_ratio)

        resized = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

        denoised = cv2.fastNlMeansDenoising(resized, None, 10, 7, 21)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        padding = 10
        final_image = cv2.copyMakeBorder(enhanced, padding, padding, padding, padding,
                                         cv2.BORDER_CONSTANT, value=[255, 255, 255])

        final_image = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)

        return final_image

    def normalize_plate_text(self, text):
        """Заменяет латинские буквы на визуально похожие кириллические"""

        latin_to_cyrillic = {
            "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М",
            "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т",
            "Y": "У", "X": "Х"
        }
        return "".join(latin_to_cyrillic.get(c, c) for c in text)

    def correct_plate_number(self, plate):
        """Исправляет ошибки OCR на основе позиции символа (буква/цифра)"""

        plate = re.sub(r'[^a-zA-Zа-яА-Я0-9]', '', plate)

        if 8 <= len(plate) <= 9:
            if len(plate) == 9 and plate[:3].isdigit() and not plate[3].isdigit():
                plate = plate[3:] + plate[:3]

            elif len(plate) == 8 and plate[:2].isdigit() and not plate[2].isdigit():
                plate = plate[2:] + plate[:2]

        plate = plate.upper()
        plate_list = list(plate)

        target_positions_liter = [0, 4, 5]
        target_positions_number = [1, 2, 3, 6, 7, 8]

        replacements_liter = {
            '8': 'B',
            '0': 'O',
            'V': 'Y',
            '1': 'T'
        }

        replacements_number = {
            'O': '0', 
            'I': '1', 
            'Q': '0', 
            'D': '0', 
            'B': '8',
            'S': '5',
        }

        for pos in target_positions_liter:
            if pos < len(plate_list):
                char = plate_list[pos]
                if char in replacements_liter:
                    plate_list[pos] = replacements_liter[char]

        for pos in target_positions_number:
            if pos < len(plate_list):
                char = plate_list[pos]
                if char in replacements_number:
                    plate_list[pos] = replacements_number[char]

        return ''.join(plate_list)

    def is_license_plate(self, text):
        """Проверяет валидность номера и возвращает нормализованный вид"""

        plate_text = self.normalize_plate_text(text)

        is_valid = bool(self.license_pattern.match(plate_text))

        return is_valid, plate_text

    def recognize(self, img):
        """Запускает полный цикл распознавания, коррекциии и валидации номера"""

        processed_img = self.preprocess_for_ocr(img)
        result = self.ocr.predict(processed_img)
        if result and result[0]['rec_texts']:
            plate = ''.join(line for line in result[0]['rec_texts'])
        else:
            return None

        plate = self.correct_plate_number(plate)
        valid, normalized_plate = self.is_license_plate(plate)

        if valid:
            return normalized_plate
        return None
