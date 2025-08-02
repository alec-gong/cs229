import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete p01b_logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    plot_path = save_path.replace('.txt', '.png')
    plot_path_true = plot_path.replace(WILDCARD, 'true')
    plot_path_naive = plot_path.replace(WILDCARD, 'naive')
    plot_path_adjusted = plot_path.replace(WILDCARD, 'adjusted')

    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)

    lg = LogisticRegression()
    lg.fit(x_train, t_train)
    y_predict = lg.predict(x_test)
    np.savetxt(output_path_true, y_predict)
    util.plot(x_test, t_test, lg.theta, plot_path_true)

    # Part (b): Train on y-labels and test on true labels
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)

    lg = LogisticRegression()
    lg.fit(x_train, y_train)
    y_predict = lg.predict(x_test)
    np.savetxt(output_path_naive, y_predict)
    util.plot(x_test, t_test, lg.theta, plot_path_naive)

    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    x_val, y_val = util.load_dataset(valid_path, label_col='y')
    x_val = x_val[y_val==1, :]
    x_val = util.add_intercept(x_val)
    y_pred = lg.predict(x_val)
    alpha = np.mean(y_pred)
    print("alpha is ", alpha)
    np.savetxt(output_path_adjusted,  y_predict/ alpha)
    util.plot(x_test, t_test, lg.theta, plot_path_adjusted, correction=alpha)
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
