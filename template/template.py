# from dust i have come, dust i will be

number_of_features = 0
number_of_classes = 0
dataset_size = 0


class Object:
    def __init__(self, class_name):
        self.class_name = class_name
        self.features = []


Object_Dictionary = {}


def read_dataset():
    global number_of_features, number_of_classes, dataset_size, Object_Dictionary

    f = open("dataset.txt", "r")
    lines = f.readlines()

    number_of_features, number_of_classes, dataset_size = map(int, lines[0].split())

    for i in range(dataset_size):
        data = lines[i + 1].split()

        class_name = data[number_of_features]

        if class_name not in Object_Dictionary:
            Object_Dictionary[class_name] = Object(class_name)

        Object_Dictionary[class_name].features.append(data[: number_of_features])


if __name__ == "__main__":
    read_dataset()

    for key in Object_Dictionary:
        print(Object_Dictionary[key].class_name)
        for i in range(len(Object_Dictionary[key].features)):
            print(Object_Dictionary[key].features[i])
