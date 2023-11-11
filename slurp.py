from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ZIP_CODE_HEADER = "What is your ZIP code?"
GENDER_HEADER = "Gender"
FAVORITE_COFFEE = "Lastly, what was your favorite overall coffee?"
NUMBER_OF_CHILDREN = "Number of Children"
POLITICS = "Political Affiliation"
EDUCATION_LEVEL = "Education Level"
HOUSEHOLD_INCOME = "Household Income"
AGE_HEADER = "What is your age?"


def drop_in_place(df: pd.DataFrame, column: str) -> None:
    df.drop(column, axis=1, inplace=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    drop_in_place(df, "Respondent ID")
    drop_in_place(df, "Submission ID")
    drop_in_place(df, "Submitted at")
    drop_in_place(df, "Where do you typically drink coffee?")
    drop_in_place(df, "How do you brew coffee at home?")
    drop_in_place(df, "On the go, where do you typically purchase coffee?")
    drop_in_place(df, "Do you usually add anything to your coffee?")
    drop_in_place(df, "What kind of dairy do you add?")
    drop_in_place(df, "What kind of sugar or sweetener do you add?")
    drop_in_place(df, "What kind of flavorings do you add?")
    drop_in_place(df, "Coffee A - Notes")
    drop_in_place(df, "Coffee B - Notes")
    drop_in_place(df, "Coffee C - Notes")
    drop_in_place(df, "Coffee D - Notes")
    drop_in_place(df, "Why do you drink coffee?")
    drop_in_place(df, "Ethnicity/Race (please specify)")
    drop_in_place(df, "Gender (please specify)")
    drop_in_place(df, "Other reason for drinking coffee")
    drop_in_place(df, "What else do you add to your coffee?")
    drop_in_place(df, "Please specify what your favorite coffee drink is")
    drop_in_place(df, "Where else do you purchase coffee?")
    drop_in_place(df, "How else do you brew coffee at home?")
    drop_in_place(df, ZIP_CODE_HEADER)
    return df


def init_dataframe() -> pd.DataFrame:
    with open("tmp/GACTT_RESULTS_ANONYMIZED.csv", newline="") as f:
        df = pd.read_csv(f)
        return clean(df)


def single_dimension_report(df: pd.DataFrame) -> None:
    # TODO: were there any duplicates?
    headers = df.columns.unique()
    for header in headers:
        count = df[header].count()
        print(f"> Single-dimension report for {header}: {count}")
        unique_values = df[header].unique()
        for unique_value in unique_values:
            sub_count = df[df[header] == unique_value][header].count()
            print(f"  - {unique_value}: {sub_count}")


def frequency_single(df: pd.DataFrame, column: str) -> Tuple[List[Any], List[int]]:
    frequency: Dict[int, int] = {}
    values = df[column].unique()
    for value in values:
        print(value)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue

        value_count = df[df[column] == value][column].count()
        if value not in frequency:
            frequency[value] = 0
        frequency[value] += value_count

    values = []
    counts = []
    for val, count in frequency.items():
        values.append(val)
        counts.append(count)
    return values, counts


def display_pie_chart(df: pd.DataFrame, column: str) -> None:
    labels, counts = frequency_single(df, column)
    plt.pie(counts, labels=labels, autopct="%1.1f%%")
    plt.title(column)
    plt.tight_layout()
    plt.show()


def main():
    df = init_dataframe()

    print(df.columns.unique())

    single_dimension_report(df)

    # TODO: somehow demonstrate the scale of each one -- are there supposed to be 0s?
    display_pie_chart(df, GENDER_HEADER)
    display_pie_chart(df, FAVORITE_COFFEE)
    display_pie_chart(df, NUMBER_OF_CHILDREN)
    display_pie_chart(df, HOUSEHOLD_INCOME)
    display_pie_chart(df, POLITICS)
    display_pie_chart(df, EDUCATION_LEVEL)
    display_pie_chart(df, AGE_HEADER)


#  print(df)

# single_dimension_report(df)


if __name__ == "__main__":
    main()
