import csv
import sys
import pandas as pd


def init_dataframe() -> pd.DataFrame:
    with open("tmp/GACTT_RESULTS_ANONYMIZED.csv", newline="") as f:
        df = pd.read_csv(f)

        return df


# TODO sanitize the data


def single_dimension_report(df: pd.DataFrame) -> None:
    # TODO: were there any duplicates?
    df.drop("Respondent ID", axis=1, inplace=True)
    df.drop("Submission ID", axis=1, inplace=True)
    df.drop("Submitted at", axis=1, inplace=True)
    headers = df.columns.unique()
    for header in headers:
        count = df[header].count()
        print(f"> Single-dimension report for {header}: {count}")
        unique_values = df[header].unique()
        for unique_value in unique_values:
            sub_count = df[df[header] == unique_value][header].count()
            print(f"  - {unique_value}: {sub_count}")


def main():
    df = init_dataframe()
    print(df.columns.unique())

    print(df["Where do you typically drink coffee? (At home)"])

    single_dimension_report(df)


#  print(df)

# single_dimension_report(df)


if __name__ == "__main__":
    main()
