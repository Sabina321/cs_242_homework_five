import numpy
from matplotlib import pylab

class Perceptron(object):
    def __init__(self):
        self.dataset = None
        self.w = numpy.zeros(15)
        self.taught = False
        self.weights = []

    def two_epoch_run(self):
        for i in range(2):
            for element in self.dataset:
                prediction = numpy.sign(numpy.dot(self.w, element["data"]))

                if prediction != element["label"]:
                    self.w += numpy.dot(element["label"], element["data"])

                self.weights.append(self.w)

    def last_hypothesis(self, test_dataset):
        error_count = 0
        this_w = self.weights[-1]
        for element in test_dataset:
            if element["label"] != numpy.sign(numpy.dot(this_w, element["data"])):
                error_count += 1

        return error_count

    def average_hypothesis(self, test_dataset):
        error_count = 0
        this_w = numpy.sum(self.weights, axis=0)/1000.
        for element in test_dataset:
            if element["label"] != numpy.sign(numpy.dot(this_w, element["data"])):
                error_count += 1

        return error_count

    def voted_hypothesis(self, test_dataset):
        error_count = 0
        for element in test_dataset:
            plus_counts = 0
            for w in self.weights:
                if numpy.sign(numpy.dot(w, element["data"])) > 0:
                    plus_counts += 1

            if plus_counts == 500:
                error_count += 1
            elif plus_counts > 500 and element["label"] < 0:
                error_count += 1
            elif plus_counts < 500 and element["label"] > 0:
                error_count += 1

        return error_count

    def last_epoch_average(self, test_dataset):
        error_count = 0
        this_w = numpy.sum(self.weights[499:], axis=0)/500.
        for element in test_dataset:
            if element["label"] != numpy.sign(numpy.dot(this_w, element["data"])):
                error_count += 1

        return error_count

    def last_epoch_vote(self, test_dataset):
        error_count = 0
        for element in test_dataset:
            sum = 0
            for w in self.weights[499:]:
                sum += numpy.sign(numpy.dot(w, element["data"]))

            vote = numpy.sign(sum)
            if vote != element["label"]:
                error_count += 1

        return error_count

    def train(self):
        """ learn over the dataset """
        # reset the dataset each time
        self.w = numpy.zeros(15)
        self.epochs = 0
        self.errors = []
        self.done = False

        while not self.done:
            error_count = 0

            for element in self.dataset:
                prediction = numpy.sign(numpy.dot(self.w, element["data"]))

                if prediction != element["label"]:
                    error_count += 1
                    self.w += numpy.dot(element["label"], element["data"])

            self.errors.append(error_count)
            self.epochs += 1

            if error_count == 0 or self.epochs > 1000:
                self.done = True

        return self.errors


    def consume_dataset(self, dataset):
        self.dataset = dataset


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
                element["label"] = 1
            else:
                element["label"] = -1

    def generate_b(self):
        """ if at least seven of the features are +1, we label with +1,
        else, we label with -1"""
        for element in self.dataset:
            if sum(element["data"]) >= -1:
                element["label"] = 1
            else:
                element["label"] = -1

    def generate_c(self):
        for element in self.dataset:
            randomness = 8*numpy.random.random() - 4
            label = randomness + sum(element["data"][:11])
            if label < 0:
                element["label"] = -1
            else:
                element["label"] = 1

def a(dg):

    dg.generate_a()

    prcp = Perceptron()
    prcp.consume_dataset(dg.dataset)
    prcp.train()

    print prcp.epochs

    print prcp.errors

    print prcp.w


def b(dg):

    dg.generate_b()

    prcp = Perceptron()
    prcp.consume_dataset(dg.dataset)
    error = prcp.train()

    pylab.plot(error)
    pylab.xlabel("Epoch")
    pylab.ylabel("Error count")
    pylab.savefig("prcp.pdf")
    print prcp.epochs

    print prcp.errors


def c(dg):
    dg.generate_c()

    prcp = Perceptron()
    prcp.consume_dataset(dg.dataset)
    prcp.two_epoch_run()

    test_set = DataGenerator()
    test_set.generate_dataset()
    test_set.generate_c()

    print "Last hypothesis"
    print prcp.last_hypothesis(test_set.dataset)

    print "Average hypothesis"
    print prcp.average_hypothesis(test_set.dataset)

    print "Voted hypothesis"
    print prcp.voted_hypothesis(test_set.dataset)

    print "Last epoch average"
    print prcp.last_epoch_average(test_set.dataset)

    print "Last epoch vote"
    print prcp.last_epoch_vote(test_set.dataset)


if __name__ == '__main__':
    dg = DataGenerator()
    dg.generate_dataset()

    # a(dg)

    b(dg)

    # c(dg)