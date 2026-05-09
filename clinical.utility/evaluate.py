import pandas as pd
import numpy as np
from glob import glob

def format_val(pct, err):
    pct_str = str(pct)
    if pct_str.endswith(".0"):
        pct_str = pct_str[:-2]
    return f"{pct_str} ({err})"

if __name__ == '__main__':
    data_records = {}

    for c in glob("./results/*_eval.csv"):
        center_name = c.split("/")[2].split("_")[0]
        main_df = pd.read_csv(c)
        raters = [x.replace("Stage1_", "") for x in main_df.keys() if "Stage1_"  in x]
        for rater_name in raters:
            df = main_df[["path", f"Stage1_{rater_name}", f"Stage2_{rater_name}"]].copy()

            for stage in ["Stage1", "Stage2"]:
                acc = df[f"{stage}_{rater_name}"].values
                pct = np.round(100 * (len(acc) - np.sum(acc)) / len(acc), 1)

                row_key = f"{center_name}_{stage}"

                if row_key not in data_records:
                    data_records[row_key] = {}

                data_records[row_key][rater_name] = format_val(pct, int(np.sum(acc)))

    summary_df = pd.DataFrame.from_dict(data_records, orient='index').sort_index()
    print(summary_df)
