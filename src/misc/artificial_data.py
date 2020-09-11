import numpy as np


def artificial_task1(
        input_dim: int = 2,
        num_train_examples: int = 10000,
        num_tasks: int = 5,
        seed: int = 0,
):
    # blackboxes are quadratic functions whose centers are sampled in a ball around [0.5, ..., 0.5]
    np.random.seed(seed)
    centers = (np.random.rand(num_tasks, input_dim) - 0.5) * 0.25 + 0.5
    Xys = []
    for x_star in centers:
        X = np.random.rand(num_train_examples, input_dim)
        y = np.square((X - x_star)).mean(axis=-1, keepdims=True)
        Xys.append((X, y))
    Xy_train = Xys[1:]
    X_test, y_test = Xys[0]

    return Xy_train, X_test, y_test


def artificial_task2(
        input_dim: int = 2,
        num_train_examples: int = 10000,
        num_tasks: int = 5,
        seed: int = 0,
):
    # blackboxes are quadratic functions whose centers are either [0.25, ..., 0.25] or [0.75, ..., 0.75]
    # this is a tasks that requires adaptation so that TS should be as good as RS and outperformed by GP and GP3
    # GP2 and GP3 should have the same performance
    np.random.seed(seed)
    sign = 2 * (np.random.randint(low=0, high=2, size=num_tasks) - 0.5)
    # the first sign is set to 1 so that there is prior knowledge
    sign[0] = 1
    center = np.ones(input_dim) * 0.5
    shift = 0.25 * (np.ones(input_dim).reshape(1, -1) * sign.reshape(-1, 1))
    centers = center + shift
    Xys = []
    for x_star in centers:
        X = np.random.rand(num_train_examples, input_dim)
        y = np.square((X - x_star)).mean(axis=-1, keepdims=True)
        Xys.append((X, y))
    Xy_train = Xys[1:]
    X_test, y_test = Xys[0]

    return Xy_train, X_test, y_test


if __name__ == '__main__':
    artificial_task2()