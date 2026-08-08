import torch


def compute_pos_weight(
    dataframe,
):

    label_columns = dataframe.columns[
        1:-1
    ]

    total_samples = len(
        dataframe,
    )

    pos_weights = []

    for column in label_columns:

        positives = dataframe[
            column
        ].sum()

        negatives = (
            total_samples
            - positives
        )

        weight = negatives / positives

        pos_weights.append(
            weight,
        )

    return torch.tensor(
        pos_weights,
        dtype=torch.float32,
    )