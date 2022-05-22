#@title Dataset Creation Helper Functions
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class AttributesDataset():
    def __init__(self, annotation_path):
        Clases_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                Clases_labels.append(row['Clase'])

        self.Clases_labels = np.unique(Clases_labels)
        self.num_Clases = len(self.Clases_labels)

        self.Clase_id_to_name = dict(zip(range(len(self.Clases_labels)), self.Clases_labels))
        self.Clase_name_to_id = dict(zip(self.Clases_labels, range(len(self.Clases_labels))))

class PotatoDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.Clases_labels = []

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                self.Clases_labels.append(self.attr.Clase_name_to_id[row['Clase']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]

        # read image
        img = Image.open(img_path)

        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)

        # return the image and all the associated labels
        img = img
        Clases = self.Clases_labels[idx]

        return img, Clases