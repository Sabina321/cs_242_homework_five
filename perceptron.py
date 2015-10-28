import numpy, matplotlib

class Perceptron(object):
    def __init__(self):
        pass

    def __call__(self):
        pass


class DataGenerator(object):
    def __init__(self, instance_count=500):
        self.count = instance_count
        self.dataset = []

    def __call__(self, option="a"):
        if self.dataset is []:
            self.generate_dataset()
        if option == "a":
            return self.generate_a()
        elif option == "b":
            return self.generate_b()
        elif option == "c":
            return self.generate_c()

    def generate_dataset(self):
        for i in range(self.count):
            new_point = []
            for j in range(15):
                if numpy.random.random() < 0.5:
                    new_point.append(-1)
                else:
                    new_point.append(1)
            self.dataset.append({
                "data" : new_point,
                "label" : None
            })

    def generate_a(self):
        for element in self.dataset:
            if element["data"][0] > 0:
                element["label"] = "+"
            else:
                element["label"] = "-"

    def generate_b(self):
        """ if at least seven of the features are +1, we label with +1,
        else, we label with -1"""
        for element in self.dataset:
            if sum(element["data"]) >= -1:
                element["label"] = "+"
            else:
                element["label"] = "-"

    def generate_c(self):
        for element in self.dataset:
            randomness = 8*numpy.random.random() - 4
            label = randomness + sum(element["data"][:11])
            if label < 0:
                element["label"] = "-"
            else:
                element["label"] = "+"


if __name__ == '__main__':
    dg = DataGenerator()
    dg.generate_dataset()
    print dg.dataset