import numpy
import scipy.special
import time
import matplotlib.pyplot


class NeuralNetwork:
    # INITIALIZE neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #  set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        #   w11 w22
        #   w12 w22

        # Initial random weights simple approach
        # self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.wih = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        # more sophisticated initial weights based off the std of the size of the network
        self.wih = numpy.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)
        )
        self.who = numpy.random.normal(
            0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)
        )

        # Learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    # TRAIN the neural network
    def train(self, input_list, target_list):
        # convert inputs lst to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # claculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emering form final output layer
        final_outputs = self.activation_function(final_inputs)

        # ERRORS
        # error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # UPDATE WEIGHTS
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs),
        )
        # updates the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs),
        )

    # QUERY the neural network
    def query(self, inputs_list):
        # INPUTS
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # HIDDEN LAYER
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # OUTPUT LAYER
        # calculate the signals into the final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


def train_network(network, training_data, output_layer):
    for record in training_data:
        # split the record by the "," commans
        all_values = record.split(",")

        # scale and shift the inputs
        input_values = numpy.asarray(all_values[1:]).astype(numpy.float32)
        inputs = (input_values / 255.0 * 0.99) + 0.01

        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_layer) + 0.01

        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        network.train(inputs, targets)


def test_network(network, test_data, scores):
    # go through all the records in the test data set
    for record in test_data:
        # split the record by the "," commas
        all_values = record.split(",")
        # correct answer is first value
        correct_label = int(all_values[0])
        # print(correct_label, "correct label")
        # scale and shift the inputs
        input = numpy.asarray(all_values[1:]).astype(numpy.float32)
        inputs = (input / 255.00 * 0.99) + 0.01
        # query the network
        outputs = network.query(inputs)
        #  the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        # print(label, "networks answer")
        # append correct or incorrect to list
        if label == correct_label:
            # network's answer matches correct answer, add 1 to scorecard
            scores.append(1)
        else:
            # networks answer doesn't match correct answer, add 0 to scorecard
            scores.append(0)


def run_experiment(hidden_nodes, learning_rate, epochs, training_data, test_data):
    """
    runs a neural network training and testing experiment with given hyperparameters
    returns findal accuracy
    """
    print("---Running Experiemnt ---")
    print(
        f"Config: LR={learning_rate:.2f}, Epochs={epochs}, HiddenNodes={hidden_nodes}"
    )

    input_nodes = 784
    output_nodes = 10
    start_time = time.time()

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # training
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}...")
        train_network(nn, training_data, output_nodes)

    # test
    scorecard = []
    test_network(nn, test_data, scorecard)

    scorecard_array = numpy.asarray(scorecard)
    performance = scorecard_array.sum() / scorecard_array.size
    end_time = time.time()

    print(f"Performance: {performance:.4f} (took {end_time - start_time:.2f} seconds)")
    return performance


def plot_performance(x, y, title):
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.plot(x, y, marker="o")
    matplotlib.pyplot.title(f"NN performance vs. {title}")
    matplotlib.pyplot.xlabel(title)
    matplotlib.pyplot.xlabel("Accuracy")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.xticks(x)
    matplotlib.pyplot.show()
