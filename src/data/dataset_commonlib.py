import pandas as pd

def read_csv(file: str = "../data/processed/Sevilla_2015.csv") -> pd.DataFrame:
    df = pd.read_csv("../data/processed/Sevilla_2015.csv")
    df.head()

    df["date"] = pd.to_datetime(df["date"], format='%d-%b-%Y %H:%M:%S')
    df.set_index("date")

    return df

def concatenate_dataframes(*tables: pd.DataFrame):
    result = pd.concat(tables)
