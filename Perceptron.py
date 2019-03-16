import numpy
from math import exp
from math import ceil
from sklearn.utils import resample
from sklearn.utils import shuffle
import copy
import matplotlib.pyplot as plt
import time


max_epoch = 300
validation_epoch = 30


def compute_activation(init_array):
    row_length = len(init_array)
    col_length = len(init_array[0])
    for i in range(row_length):
        for j in range(col_length):
            init_array[i][j] = 1 / (1 + exp(-1 * init_array[i][j]))
    return init_array


def create_label_matrix(labels):
    num_of_classes = len(numpy.unique(numpy.array(labels)))
    size = (len(labels), num_of_classes)
    labels_len = len(labels)
    label_matrix = numpy.zeros(size)
    for i in range(0, labels_len):
        cl = int(labels[i])
        index = cl - 1
        label_matrix[i][index] = 1

    return label_matrix


def get_training_and_validation_set(filename, percentage_of_training_set):
    with open(filename, 'r') as f:
        size_training_and_validation_set = len(f.readlines())
    with open(filename, 'r') as f:
        num_of_columns = len(f.readline().split(","))

    size_of_training_set = ceil(size_training_and_validation_set * percentage_of_training_set)
    size_of_validation_set = size_training_and_validation_set - size_of_training_set

    training_set_matrix = numpy.zeros((size_of_training_set, num_of_columns), dtype=float)
    validation_set_matrix = numpy.zeros((size_of_validation_set, num_of_columns), dtype=float)

    with open(filename, 'r') as f:
        for i in range(0, size_of_training_set):
            array_of_values = f.readline().split(",")
            len_array_of_values = len(array_of_values)
            for j in range(0, len_array_of_values):
                training_set_matrix[i][j] = array_of_values[j]

        for i in range(0, size_of_validation_set):
            array_of_values = f.readline().split(",")
            len_array_of_values = len(array_of_values)
            for j in range(0, len_array_of_values):
                validation_set_matrix[i][j] = array_of_values[j]

    with open("validation_set.csv", 'w') as file:
        for row in range(size_of_validation_set):
            for col in range(num_of_columns):
                file.write(str(validation_set_matrix[row][col]))
                if col == num_of_columns - 1:
                    break
                if col != num_of_columns:
                    file.write(",")
            file.write("\n")

    num_of_features = len(training_set_matrix[0])
    return training_set_matrix, validation_set_matrix, size_of_training_set, size_of_validation_set, num_of_features


def create_labels_matrix_train_validation(filename, size_of_training, size_of_validation):
    validation_labels = numpy.zeros(size_of_validation, dtype=int)
    training_labels = numpy.zeros(size_of_training, dtype=int)
    with open(filename, 'r') as f:
        for i in range(0, size_of_training):
            training_labels[i] = int(f.readline())

        for i in range(0, size_of_validation):
            validation_labels[i] = int(f.readline())

    with open("validation_labels.csv", 'w') as file:
        for row in range(size_of_validation):
            file.write(str(validation_labels[row]))
            file.write("\n")

    validation_labels_matrix = create_label_matrix(validation_labels)
    num_of_classes = len(numpy.unique(validation_labels))
    return validation_labels_matrix, num_of_classes, training_labels


def get_test_set(filename):
    with open(filename, 'r') as f:
        size_test_set = len(f.readlines())
    with open(filename, 'r') as f:
        num_of_columns = len(f.readline().split(","))

    test_set_mat = numpy.zeros((size_test_set, num_of_columns), dtype=float)

    with open(filename, 'r') as f:
        for i in range(0, size_test_set):
            array_of_values = f.readline().split(",")
            len_array_of_values = len(array_of_values)
            for j in range(0, len_array_of_values):
                test_set_mat[i][j] = array_of_values[j]

    return test_set_mat


def write_predicted_array(filename, predictions):
    len_predictions = len(predictions)
    with open(filename, 'w') as f:
        for i in range(0, len_predictions):
            f.write(str(predictions[i]))
            f.write("\n")
    print("Predictions written at predicted_ann.csv.")


def create_upsampled_test_data(dataset, labels, num_of_classes):
    len_dataset = len(dataset)

    freq_array = [0 for x in range(num_of_classes)]
    new_label = numpy.reshape(labels, (len_dataset, 1))

    for label in labels:
        freq_array[label - 1] += 1

    concat_data_label = numpy.concatenate((dataset, new_label), axis=1)
    max_no_of_class = max(freq_array)
    index_of_max_class = numpy.argmax(freq_array)
    len_concat_data_label = len(concat_data_label[0])

    # upsample
    for label in range(num_of_classes):
        new_array = numpy.zeros((freq_array[label], len_concat_data_label))
        index = 0
        for count in range(0, len_dataset):
            if label == labels[count] - 1:
                new_array[index] = concat_data_label[count]
                index += 1
        if label == index_of_max_class:
            upsamped = copy.deepcopy(new_array)
        else:
            upsamped = resample(copy.deepcopy(new_array), n_samples=max_no_of_class)

        if label == 0:
            resampled_arr = copy.deepcopy(upsamped)
        else:
            resampled_arr = numpy.concatenate((resampled_arr, upsamped), axis=0)

    resampled_arr = shuffle(resampled_arr)
    len_upsampled_col = len(resampled_arr[0])
    len_upsampled_row = len(resampled_arr)
    labels_upsampled = numpy.array(resampled_arr[:, [len_upsampled_col - 1]])
    resampled_arr = numpy.delete(resampled_arr, [len_upsampled_col - 1], axis=1)
    len_upsampled_col -= 1

    with open("training_set.csv", 'w') as file:
        for row in range(len_upsampled_row):
            for col in range(len_upsampled_col):
                file.write(str(resampled_arr[row][col]))
                if col == len_upsampled_col - 1:
                    break
                if col != len_upsampled_col:
                    file.write(",")
            file.write("\n")

    with open("training_labels.csv", 'w') as file:
        for row in range(len_upsampled_row):
            file.write(str(labels_upsampled[row][0]))
            file.write("\n")

    labels_upsampled = numpy.reshape(labels_upsampled, len(labels_upsampled))
    labels_matrix_upsampled = create_label_matrix(labels_upsampled)

    return resampled_arr, labels_matrix_upsampled


class NeutralNetwork:
    learning_rate = 0.1

    def __init__(self, input_no, layer_1, layer_2, output_no, eta):
        self.number_of_inputs = input_no
        self.number_of_layer_1 = layer_1
        self.number_of_layer_2 = layer_2
        self.number_of_outputs = output_no
        self.learning_rate = eta

        self.input_layer = numpy.zeros((self.number_of_inputs, 1))
        self.layer_one_weights = numpy.random.uniform(-0.1, 0.1, size=(self.number_of_layer_1, self.number_of_inputs))
        self.layer_one_bias = numpy.random.uniform(-0.1, 0.1, size=(self.number_of_layer_1, 1))
        self.layer_two_weights = numpy.random.uniform(-0.1, 0.1, size=(self.number_of_layer_2, self.number_of_layer_1))
        self.layer_two_bias = numpy.random.uniform(-0.1, 0.1, size=(self.number_of_layer_2, 1))
        self.output_layer_weights = numpy.random.uniform(-0.1, 0.1, size=(self.number_of_outputs,
                                                                          self.number_of_layer_2))
        self.output_layer_bias = numpy.random.uniform(-0.1, 0.1, size=(self.number_of_outputs, 1))

        self.output_layer = numpy.zeros((self.number_of_outputs, 1))

    def validate(self, dataset, labels):
        dataset_len = len(dataset)
        correct_prediction = 0
        for i in range(0, dataset_len):
            x_in = numpy.reshape(dataset[i], (len(dataset[i]), 1))
            d_out = numpy.reshape(labels[i], (len(labels[i]), 1))
            # forward propagation

            # 1st hidden layer
            v_hidden_layer_1 = numpy.dot(self.layer_one_weights, x_in) + self.layer_one_bias
            y_hidden_layer_1 = compute_activation(v_hidden_layer_1)

            # 2nd hidden layer
            v_hidden_layer_2 = numpy.dot(self.layer_two_weights, y_hidden_layer_1) + self.layer_two_bias
            y_hidden_layer_2 = compute_activation(v_hidden_layer_2)

            # output layer
            v_output_layer = numpy.dot(self.output_layer_weights, y_hidden_layer_2) + self.output_layer_bias
            # final_output = compute_activation(v_output_layer)
            final_output = compute_activation(v_output_layer)

            predicted_label = numpy.argmax(final_output)
            true_label = numpy.argmax(numpy.array(labels[i]))

            if predicted_label == true_label:
                correct_prediction += 1

            error_vector = d_out - final_output

        err = numpy.multiply(error_vector, error_vector)
        err = numpy.divide(err, 2)
        error_sum = numpy.sum(err)
        accuracy = correct_prediction / dataset_len
        print("Error: {} Accuracy: {}".format(error_sum, accuracy))

        return error_sum, accuracy

    def test(self, dataset):
        print("Testing phase...")
        dataset_len = len(dataset)
        labels = numpy.zeros(dataset_len)
        for i in range(0, dataset_len):
            x_in = numpy.reshape(dataset[i], (len(dataset[i]), 1))

            # 1st hidden layer
            v_hidden_layer_1 = numpy.dot(self.layer_one_weights, x_in) + self.layer_one_bias
            y_hidden_layer_1 = compute_activation(v_hidden_layer_1)

            # 2nd hidden layer
            v_hidden_layer_2 = numpy.dot(self.layer_two_weights, y_hidden_layer_1) + self.layer_two_bias
            y_hidden_layer_2 = compute_activation(v_hidden_layer_2)

            # output layer
            v_output_layer = numpy.dot(self.output_layer_weights, y_hidden_layer_2) + self.output_layer_bias
            final_output = compute_activation(v_output_layer)

            # Get error (squared error)
            label = numpy.argmax(final_output) + 1
            labels[i] = label

        return labels

    def train(self, test_set, test_labels, validation_set, validation_label):
        print("Training phase...")
        total_error = numpy.zeros(max_epoch)
        validation_acc = numpy.zeros(ceil(max_epoch / validation_epoch))
        validation_err = numpy.zeros(ceil(max_epoch / validation_epoch))
        v_e = 0
        size_of_traning_set = len(test_set)
        len_test_set_col = len(test_set[0])
        len_test_label_col = len(test_labels[0])

        base_time = time.time()
        validation_time = 0
        for count in range(0, max_epoch):

            random_permutations = numpy.random.permutation(size_of_traning_set)
            for count_2 in range(0, size_of_traning_set):
                random_index = random_permutations[count_2]
                # random_index = count_2
                x_in = numpy.reshape(test_set[random_index], (len_test_set_col, 1))
                d_out = numpy.reshape(test_labels[random_index], (len_test_label_col, 1))
                # forward propagation

                # 1st hidden layer
                v_hidden_layer_1 = numpy.add(numpy.dot(self.layer_one_weights, x_in), self.layer_one_bias)
                y_hidden_layer_1 = compute_activation(v_hidden_layer_1)

                # 2nd hidden layer
                v_hidden_layer_2 = numpy.add(numpy.dot(self.layer_two_weights, y_hidden_layer_1), self.layer_two_bias)
                y_hidden_layer_2 = compute_activation(v_hidden_layer_2)

                v_output_layer = numpy.add(numpy.dot(self.output_layer_weights, y_hidden_layer_2), self.output_layer_bias)
                final_output = compute_activation(v_output_layer)
                error_vector = d_out - final_output

                # back propagation
                # compute gradient in output layer
                delta_output_x = numpy.multiply(error_vector, final_output)
                one_minus_out = 1 - final_output
                delta_output = numpy.multiply(delta_output_x, one_minus_out)

                # compute gradient in hidden layer 2
                one_minus_y_h2 = 1 - y_hidden_layer_2
                output_layer_weights_trans = numpy.transpose(self.output_layer_weights)
                deriv_hidden_layer_2_x = numpy.multiply(y_hidden_layer_2, one_minus_y_h2)
                deriv_out_layer = numpy.dot(output_layer_weights_trans, delta_output)
                delta_hidden_layer_2 = numpy.multiply(deriv_hidden_layer_2_x, deriv_out_layer)

                # compute gradient in hidden layer 1
                one_minus_y_h1 = 1 - y_hidden_layer_1
                hidden_layer_2_weights_trans = numpy.transpose(self.layer_two_weights)
                deriv_hidden_layer_1_x = numpy.multiply(y_hidden_layer_1, one_minus_y_h1)
                deriv_layer_2 = numpy.dot(hidden_layer_2_weights_trans, delta_hidden_layer_2)
                delta_hidden_layer_1 = numpy.multiply(deriv_hidden_layer_1_x, deriv_layer_2)

                # update weights and biases of output layer
                self.output_layer_weights = self.output_layer_weights + \
                                            numpy.multiply(self.learning_rate, numpy.dot(delta_output,
                                                      numpy.reshape(y_hidden_layer_2, (1, self.number_of_layer_2))))
                self.output_layer_bias = self.output_layer_bias + numpy.multiply(self.learning_rate, delta_output)

                # update weights and biases of hidden layer 2
                self.layer_two_weights = self.layer_two_weights + \
                                            numpy.multiply(self.learning_rate, numpy.dot(delta_hidden_layer_2,
                                                      numpy.reshape(y_hidden_layer_1, (1, self.number_of_layer_1))))
                self.layer_two_bias = self.layer_two_bias + numpy.multiply(self.learning_rate, delta_hidden_layer_2)

                # update weights and biases of hidden layer 1
                self.layer_one_weights = self.layer_one_weights + \
                                         numpy.multiply(self.learning_rate, numpy.dot(delta_hidden_layer_1,
                                                   numpy.reshape(x_in, (1, self.number_of_inputs))))
                self.layer_one_bias = self.layer_one_bias + numpy.multiply(self.learning_rate, delta_hidden_layer_1)

            err_sum = numpy.multiply(error_vector, error_vector)
            err_sum = numpy.divide(err_sum, 2)
            total_error[count] = total_error[count] + numpy.sum(err_sum)

            print('Epoch: {} Error: {}'.format(count, total_error[count]))

            if count % validation_epoch == 0 and count != 0:
                valid_time_s = time.time()
                print("\n##################### Validation {}#####################".format(v_e))
                validation_err[v_e], validation_acc[v_e] = self.validate(validation_set, validation_label)
                print("\n")
                elapsed_valid_time = time.time() - valid_time_s
                validation_time += elapsed_valid_time
                v_e += 1

        elapsed_time = time.time() - base_time
        elapsed_time = elapsed_time - validation_time
        validation_err[v_e], validation_acc[v_e] = self.validate(validation_set, validation_label)

        print("Training duration: {} seconds".format(elapsed_time))
        print(total_error)
        print(validation_err)
        print(validation_acc)
        # show figures of training plot
        # Uncomment if you want to see plot
        # plt.figure(0)
        # plt.plot([i for i in range(max_epoch)], total_error)
        # plt.title("Training Error over {} epochs".format(max_epoch))
        #
        # plt.figure(1)
        # plt.plot([i for i in range(len(validation_err))], validation_err)
        # plt.title("Validation Error over {} validation epochs".format(len(validation_err)))
        #
        # plt.figure(2)
        # plt.plot([i for i in range(len(validation_acc))], validation_acc)
        # plt.title("Accuracy over {} validation epochs".format(len(validation_err)))
        #
        # plt.show()


print("Started ANN...")
print("Pre-processing data...")
training_set_x, validation_set_x, training_set_size, validation_set_size, no_of_features = \
    get_training_and_validation_set("data.csv", 0.8)
validation_label_mat, no_of_classes, train_label = \
    create_labels_matrix_train_validation("data_labels.csv", training_set_size, validation_set_size)
upsampled_training_set, upsampled_train_labels = create_upsampled_test_data(training_set_x, train_label, no_of_classes)
test_set_matrix = get_test_set("test_set.csv")

# Create Neural Network
input_x = no_of_features
output_x = no_of_classes
hidden_1_x = 100
hidden_2_x = 140
print("Created neural network with {} input layer, {} hidden layer 1, {} hidden layer 2 and {} output layer and"
      " 0.1 learning rate...".format(input_x, hidden_1_x, hidden_2_x, output_x))
ann = NeutralNetwork(input_x, hidden_1_x, hidden_2_x, output_x, 0.1)
ann.train(upsampled_training_set, upsampled_train_labels, validation_set_x, validation_label_mat)
predicted_labels = ann.test(test_set_matrix)
write_predicted_array("predicted_ann.csv", predicted_labels)



