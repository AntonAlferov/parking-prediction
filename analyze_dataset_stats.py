import os

import numpy as np
from PIL import Image
from tqdm import tqdm

def analyze_dataset_stats(directory):
    widths = []
    heights = []

    images = [image for image in os.listdir(directory)]

    for imagename in tqdm(images):
        path = os.path.join(directory, imagename)
        with Image.open(path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

    widths = np.array(widths)
    heights = np.array(heights)

    mean_w = np.mean(widths)
    mean_h = np.mean(heights)

    std_w = np.std(widths)
    std_h = np.std(heights)

    abs_max_w = np.max(widths)
    abs_max_h = np.max(heights)

    stat_max_w = mean_w + (3 * std_w)
    stat_max_h = mean_h + (3 * std_h)

    print('\n' + '='*40)
    print(f'СТАТИСТИКА ДАТАСЕТА: {directory}')
    print('='*40)

    print('ШИРИНА:')
    print(f'   • Среднее:           {mean_w:.2f} px')
    print(f'   • Стд. отклонение:   {std_w:.2f}')
    print(f'   • Абсолютный Макс:   {abs_max_w} px')
    print(f'   • Макс (3-сигма):    {stat_max_w:.2f} px')
    print('-' * 20)

    print('ВЫСОТА:')
    print(f'   • Среднее:           {mean_h:.2f} px')
    print(f'   • Стд. отклонение:   {std_h:.2f}')
    print(f'   • Абсолютный Макс:   {abs_max_h} px')
    print(f'   • Макс (3-сигма):    {stat_max_h:.2f} px')   
    print('='*40)