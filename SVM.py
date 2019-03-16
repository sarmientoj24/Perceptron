from sklearn import svm
import numpy
import time


def get_training_and_validation_set(filename):
    print("Reading training set...")
    with open(filename, 'r') as f:
        size_training_set = len(f.readlines())
    with open(filename, 'r') as f:
        num_of_columns = len(f.readline().split(","))

    training_set_matrix = numpy.zeros((size_training_set, num_of_columns), dtype=float)

    with open(filename, 'r') as f:
        for i in range(0, size_training_set):
            array_of_values = f.readline().split(",")
            len_array_of_values = len(array_of_values)
            for j in range(0, len_array_of_values):
                training_set_matrix[i][j] = array_of_values[j]

    return training_set_matrix, size_training_set


def get_labels(filename, size_of_training):
    print("Reading labels...")
    training_labels = numpy.zeros(size_of_training, dtype=int)
    with open(filename, 'r') as f:
        for i in range(0, size_of_training):
            entry = float(f.readline())
            entry_int = int(entry)
            training_labels[i] = entry_int

    return training_labels


def get_test_set(filename):
    with open(filename, 'r') as f:
        size_test_set = len(f.readlines())
    with open(filename, 'r') as f:
        num_of_columns = len(f.readline().split(","))

    test_set_matrix = numpy.zeros((size_test_set, num_of_columns), dtype=float)

    with open(filename, 'r') as f:
        for i in range(0, size_test_set):
            array_of_values = f.readline().split(",")
            len_array_of_values = len(array_of_values)
            for j in range(0, len_array_of_values):
                test_set_matrix[i][j] = array_of_values[j]

    return test_set_matrix


def get_test_label(filename):
    with open(filename, 'r') as f:
        size_label_set = len(f.readlines())

    test_labels = numpy.zeros(size_label_set, dtype=int)
    with open(filename, 'r') as f:
        for i in range(0, size_label_set):
            entry = float(f.readline())
            entry_int = int(entry)
            test_labels[i] = entry_int
    return test_labels


def write_predicted_array(filename, predictions):
    len_predictions = len(predictions)
    with open(filename, 'w') as f:
        for i in range(0, len_predictions):
            f.write(str(predictions[i]))
            f.write("\n")
    print("Predictions written at predicted_svm.csv.")


def process_svm(training_set, training_labels, validation_set, validation_label, test_set):
    start_time = time.time()
    print("Start training of SVM...")
    clf = svm.SVC(kernel="lieanr", max_iter=30000, gamma="auto")
    # clf = svm.NuSVC(gamma="scale", max_iter=1000000)
    clf.fit(training_set, training_labels)
    elapsed_time = time.time() - start_time
    print("Finished training after {}".format(elapsed_time))
    score = clf.score(validation_set, validation_label)
    predicted_array = clf.predict(test_set)
    print("Accuracy compared to validation data set: {} seconds".format(score))
    return predicted_array


print("SVM Testing for dataset...")
training_mat, size_of_training_set = get_training_and_validation_set("training_set.csv")
labels_array = get_labels("training_labels.csv", size_of_training_set)
validate_set_mat = get_test_set("validation_set.csv")
label_validate = get_test_label("validation_labels.csv")
test_set_mat = get_test_set("test_set.csv")
predicted = process_svm(training_mat, labels_array, validate_set_mat, label_validate, test_set_mat)
write_predicted_array("predicted_svm.csv", predicted)





