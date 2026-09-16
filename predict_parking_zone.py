import os

import torch
from torchvision import transforms


def predict_parking_zone(image, model):

    device = 'cuda:0'
    model.to(device)
    model.eval()

    ru_name_street = {
        'centralnaya5': 'Центральная д. 5',
        'centralnaya6': 'Центральная д. 6',
        'molodezhnaya18': 'Молодежная д. 18',
        'molodezhnaya20A': 'Молодежная д. 20А',
        'parkovaya52': 'Парковая д. 52',
        'parkovaya54': 'Парковая д. 54',
        'parkovaya54_52': 'Парковая д. 54-52',
        'shkolnaya3': 'Школьная д. 3',
        'shkolnaya9': 'Школьная д. 9',
        'shkolnaya11': 'Школьная д. 11',
    }

    if not os.path.exists('parking_names.txt'):
        return
    with open('parking_names.txt', 'r') as f:
        parking_names = f.read().splitlines()

    model.load_state_dict(torch.load('best_parking_space_backbone.pth'))

    state_dict_loss = torch.load('best_parking_space_centers.pth')
    class_centers = state_dict_loss['W'].to(device) 

    transform = transforms.Compose([
        transforms.Resize((1024, 576)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        embedding = model(input_tensor)

        embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=1)
        centers_norm = torch.nn.functional.normalize(class_centers, p=2, dim=0)

        similarity = torch.mm(embedding_norm, centers_norm)

        score, parking_index = torch.max(similarity, 1)

        street_name = parking_names[parking_index.item()]
        street_name_rus = ru_name_street[street_name]

        confidence = score.item()

        return street_name_rus, confidence
