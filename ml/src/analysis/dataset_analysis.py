import pandas as pd


class DatasetAnalyzer:

    def __init__(
        self,
        dataframe,
    ):

        self.dataframe = dataframe

    def analyze_class_distribution(
        self,
    ):
        label_columns = self.dataframe.columns[
            1:-1
        ]

        class_counts = {}

        for column in label_columns:

            class_counts[column] = int(
                self.dataframe[column].sum()
            )

        print("\n" + "=" * 60)
        print("TRAINING CLASS DISTRIBUTION")
        print("=" * 60)

        for class_name, count in class_counts.items():

            print(
                f"{class_name:<20} {count}"
            )

        print("=" * 60)

        print(
            f"Total Samples : {len(self.dataframe)}"
        )

        return class_counts