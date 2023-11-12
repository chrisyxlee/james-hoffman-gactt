from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

ZIP_CODE_HEADER = "What is your ZIP code?"
GENDER_HEADER = "Gender"
NUMBER_OF_CHILDREN = "Number of Children"
POLITICS = "Political Affiliation"
EDUCATION_LEVEL = "Education Level"
# TODO: fix sorting for this column
HOUSEHOLD_INCOME = "Household Income"
# TODO: fix sorting for this column
AGE_HEADER = "What is your age?"
COFFEE_LOCATION_PREFIX = "Where do you typically drink coffee?"
COFFEE_EXPERTISE = "Lastly, how would you rate your own coffee expertise?"
COFFEE_SPEND = "In total, much money do you typically spend on coffee in a month?"
COFFEE_CUPS = "How many cups of coffee do you typically drink per day?"
COFFEE_FAVORITE = "Lastly, what was your favorite overall coffee?"


def filename(s: str) -> str:
    return (
        s.replace(" ", "-")
        .replace(")", "")
        .replace("(", "")
        .replace("#", "num")
        .replace("?", "")
    )


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
        return clean(pd.read_csv(f))


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
    frequency: Dict[Any, int] = {}
    values = df[column].unique()
    for value in values:
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
    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct="%1.1f%%")
    ax.set_title(column)
    fig.savefig(f"tmp/pie_{filename(column)}", bbox_inches="tight")
    plt.show()


def to_venn3_char(row: Any, subset: List[str]) -> str:
    c0 = "1" if row[subset[0]] else "0"
    c1 = "1" if row[subset[1]] else "0"
    c2 = "1" if row[subset[2]] else "0"
    return c0 + c1 + c2


def display_venn3_diagram(df: pd.DataFrame, column_prefix: str) -> None:
    # TODO: put all subplots on the same diagram :^)
    columns = df.columns.unique()
    matched_columns = [c for c in columns if c.startswith(column_prefix)]
    print(matched_columns)

    column_values = {
        mc.removeprefix(column_prefix).strip("() "): i
        for i, mc in enumerate(matched_columns)
    }

    num_subsets = 3
    subsets = combinations(column_values, num_subsets)
    for subset in subsets:
        results = df.apply(
            lambda row: to_venn3_char(
                row, [matched_columns[column_values[s]] for s in subset]
            ),
            axis=1,  # rows
        )
        results = results.value_counts()
        for i in range(1, 7):
            s = "{0:b}".format(i)
            if s not in results:
                results[s] = 0

        fig, ax = plt.subplots()
        v = venn3(subsets=results)
        v.get_label_by_id("100").set_text(subset[0] + "\n" + str(results["100"]))
        v.get_label_by_id("010").set_text(subset[1] + "\n" + str(results["010"]))
        v.get_label_by_id("001").set_text(subset[2] + "\n" + str(results["001"]))
        ax.set_title(column_prefix)
        fig.savefig(
            f"tmp/venn_{filename(column_prefix + '=' + '-'.join(subset))}",
            bbox_inches="tight",
        )
        plt.show()


def is_safe(x: Any) -> bool:
    if x is None:
        return False

    if isinstance(x, float) and np.isnan(x):
        return False

    return True


# TODO: display normalized bar chart of relative percentages
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
    ax = plt.subplot(2, 1, 1)
    bottom = np.zeros(len(x_labels))
    for boolean, weight_count in weight_counts.items():
        ax.bar(x_labels, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title(f"{title} (raw responses)")
    ax.set_ylabel("# responses")
    # Legend on center right
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        # reverse ordering
        labelspacing=-2.5,
        frameon=False,
    )

    normalized_weight_counts = {k: v for k, v in weight_counts.items()}
    for x_val, idx in order.items():
        total = 0.0
        for y_vals in weight_counts.values():
            total += y_vals[idx]
        for y_val in weight_counts.keys():
            if total == 0:
                normalized_weight_counts[y_val][idx] = 0
            else:
                normalized_weight_counts[y_val][idx] = (
                    float(weight_counts[y_val][idx]) / float(total)
                ) * 100

    ax = plt.subplot(2, 1, 2)
    bottom = np.zeros(len(x_labels))
    for boolean, weight_count in normalized_weight_counts.items():
        ax.bar(x_labels, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count

    ax.set_title(f"{title} (normalized)")
    ax.set_ylabel("% responses")
    # Legend on center right
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        # reverse ordering
        labelspacing=-2.5,
        frameon=False,
    )

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(f"tmp/bar_{filename(title)}", bbox_inches="tight")
    plt.show()


def main():
    df = init_dataframe()

    display_venn3_diagram(df, COFFEE_LOCATION_PREFIX)

    display_stacked_bar_chart(
        df,
        column_x=AGE_HEADER,
        column_y=COFFEE_CUPS,
        title="# cups of coffee by age",
    )
    display_stacked_bar_chart(
        df,
        column_x=HOUSEHOLD_INCOME,
        column_y=COFFEE_CUPS,
        title="# cups of coffee by Household Income",
    )
    display_stacked_bar_chart(
        df,
        column_x=COFFEE_FAVORITE,
        column_y=COFFEE_EXPERTISE,
        title="Self-reported coffee expertise (1-10) by Favorite Coffee",
    )
    display_stacked_bar_chart(
        df,
        column_x=COFFEE_FAVORITE,
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

    display_pie_chart(df, GENDER_HEADER)
    display_pie_chart(df, COFFEE_FAVORITE)
    display_pie_chart(df, NUMBER_OF_CHILDREN)
    display_pie_chart(df, HOUSEHOLD_INCOME)
    display_pie_chart(df, POLITICS)
    display_pie_chart(df, EDUCATION_LEVEL)
    display_pie_chart(df, AGE_HEADER)

    single_dimension_report(df)
    print(df.columns.unique())
    return


if __name__ == "__main__":
    main()
