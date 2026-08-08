import pandas as pd

from configs import TRAIN_CSV

from src.analysis.dataset_analysis import DatasetAnalyzer


def main():

    dataframe = pd.read_csv(
        TRAIN_CSV,
    )
    print(dataframe.columns.tolist())

    analyzer = DatasetAnalyzer(
        dataframe,
    )

    analyzer.analyze_class_distribution()


if __name__ == "__main__":
    main()