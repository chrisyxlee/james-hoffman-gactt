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
COFFEE_LOCATION_PREFIX = "Where do you typically drink coffee?"
COFFEE_EXPERTISE = "Lastly, how would you rate your own coffee expertise?"
COFFEE_SPEND = "In total, much money do you typically spend on coffee in a month?"


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
    keys = list(frequency.keys())
    keys.sort()
    for key in keys:
        values.append(key)
        counts.append(frequency[key])
    return values, counts


def display_pie_chart(df: pd.DataFrame, column: str) -> None:
    labels, counts = frequency_single(df, column)
    plt.pie(counts, labels=labels, autopct="%1.1f%%")
    plt.title(column)
    plt.tight_layout()
    plt.show()


def remove_parens(s: str) -> str:
    return s.strip("() ")


def is_safe(x: Any) -> bool:
    if x is None:
        return False

    if isinstance(x, float) and np.isnan(x):
        return False

    return True


def display_stacked_bar_chart(
    df: pd.DataFrame,
    column_x: str,
    column_y: str,
    title: str,
) -> None:
    x_labels = [x for x in df[column_x].unique() if is_safe(x)]
    x_labels.sort()
    order = {x: i for i, x in enumerate(x_labels)}
    print(order)

    y_labels = [y for y in df[column_y].unique() if is_safe(y)]
    y_labels.sort()

    df = df.reset_index()  # make sure indexes pair with number of rows
    weight_counts: Dict[str, np.array] = {}
    # {
    #      "1": np.array([70, 31, 58]),
    #      "2": np.array([82, 37, 66]),
    #  }
    for y_label in y_labels:
        weight_counts[str(y_label)] = np.zeros(len(x_labels))

    for _, row in df.iterrows():
        x_val = row[column_x]
        y_val = row[column_y]

        if not is_safe(x_val) or not is_safe(y_val):
            continue

        print(f"x_val={x_val}, y_val={y_val}")
        weight_counts[str(y_val)][order[x_val]] += 1

    width = 0.5
    fig, ax = plt.subplots()
    bottom = np.zeros(len(x_labels))
    for boolean, weight_count in weight_counts.items():
        p = ax.bar(x_labels, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title(title)
    # Legend on center right
    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        # reverse ordering
        labelspacing=-2.5,
        frameon=False,
    )
    #  fig.subplots_adjust(right=0.2)
    fig.savefig("samplefigure", bbox_inches="tight")

    plt.show()


def main():
    df = init_dataframe()

    display_stacked_bar_chart(
        df,
        column_x=FAVORITE_COFFEE,
        column_y=COFFEE_EXPERTISE,
        title="Self-reported coffee expertise (1-10) by Favorite Coffee",
    )
    display_stacked_bar_chart(
        df,
        column_x=FAVORITE_COFFEE,
        column_y=GENDER_HEADER,
        title="Gender by Favorite Coffee",
    )
    display_stacked_bar_chart(
        df,
        column_x=COFFEE_EXPERTISE,
        # TODO: clean for better sorting
        column_y=HOUSEHOLD_INCOME,
        title="Self-reported coffee expertise (1-10) by Household Income",
    )

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
    return


if __name__ == "__main__":
    main()
