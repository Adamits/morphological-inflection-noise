import os
import click

import pandas as pd
import numpy as np

"""Parse the PL logs into a TSV of dev results for different languages/models/experiments.
This script steps through the assumed directory structure and aggregates the best dev performance 
for each language/model into a tsv table.

Note that by default we parse results from the most recent version, that is, the one with the
max number.

Input: results direcory, experiment id
Output: TSV of all results
"""


def make_mc_output(base_path, arch, k):
    outputs = []
    arch_path = os.path.join(base_path, arch)

    for lang in os.listdir(arch_path):
        lang_path = os.path.join(arch_path, lang)
        eval_filename = os.path.join(lang_path, "f.stats")

        last_aggregate = False
        if not os.path.isfile(eval_filename):
            print(f"WARNING: No such file {eval_filename}")
            print("Skipping...")
            continue

        with open(eval_filename, "r") as f:
            for line in f:
                if line.startswith("DEV ACCURACY (internal evaluation)"):
                    acc = line.rstrip().split("= ")[1]
                    df_dict = {k: np.nan for k in [
                        "step","train_loss", "epoch", "val_accuracy", "val_loss", "lang", "arch", "model_version"
                    ]}

                    simple_arch = arch
                    if "-" in lang:
                        dataset = "-".join(lang.split("-")[1:])
                        lang = lang.split("-")[0]
                        simple_arch = arch + "-" + dataset

                    df_dict["arch"] = simple_arch
                    df_dict["run"] = k
                    df_dict["lang"] = lang
                    df_dict["val_accuracy"] = acc
                    df = pd.DataFrame(df_dict, index=[i for i in range(len(df_dict))])
                    outputs.append(df.T)
                    break
    return outputs


def make_output(df_rows: pd.core.frame.DataFrame):
    train_loss = None
    for i, row in df_rows.iterrows():
        if "val_accuracy" not in row.keys():
            return row
        if not np.isnan(row.val_accuracy):
            output_row = row
        if not np.isnan(row.train_loss):
            train_loss = row.train_loss
    
    output_row.train_loss = train_loss
    return output_row


def get_outputs(base_path, k):
    outputs = []
    for arch in os.listdir(base_path):
        print(f"{arch}...")
        if arch.startswith("M_C"):
            outputs.extend(make_mc_output(base_path, arch, k))
            continue

        arch_path = os.path.join(base_path, arch)
        for lang in os.listdir(arch_path):
            lang_path = os.path.join(arch_path, lang)
            versions = os.listdir(lang_path)
            max_version_num = max([int(v.split("_")[-1]) for v in versions])
            file_name = os.path.join(lang_path, f"version_{max_version_num}", "metrics.csv")

            if not os.path.isfile(file_name):
                print(f"WARNING: No such file {file_name}")
                print("Skipping...")
                continue

            df = pd.read_csv(file_name)
            df["lang"] = lang
            df["arch"] = arch
            df["run"] = k
            df["model_version"] = max_version_num
            if "val_accuracy" in df.columns:
                max_val_acc_idx = df["val_accuracy"].idxmax()
                max_val_acc_epoch = df.iloc[max_val_acc_idx].epoch
                info_rows = df.loc[df["epoch"] == max_val_acc_epoch]
            else:
                info_rows = df#.loc[df["epoch"] == 1]

            outputs.append(make_output(info_rows))

    return outputs


@click.command()
@click.option("--results_dir", required=True)
@click.option("--experiment", required=True)
@click.option("--out_fn", required=True)
def main(results_dir, experiment, out_fn):
    base_path = os.path.join(results_dir, experiment)

    outputs = []
    for k in os.listdir(base_path):
        if not k.isdigit():
            continue
        k_path = os.path.join(base_path, k)
        outputs.extend(get_outputs(k_path, k))
    # if "M_C" in os.listdir(base_path):
    #     outputs = get_outputs(base_path, 1)
    # else:
    #     outputs = []
    #     for k in os.listdir(base_path):
    #         if not k.isdigit():
    #             continue
    #         k_path = os.path.join(base_path, k)
    #         outputs.extend(get_outputs(k_path), k)

    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    print("Making all one dataframe")
    out_df = pd.concat(outputs, axis=1).T
    # The way we reshape MC to fit the other dfs shape creates duplicates.
    out_df = out_df.drop_duplicates()
    print(f"Writing to {out_fn}")
    out_df[[
        "run", "step", "train_loss", "epoch", "val_accuracy", "val_loss", "lang", "arch", "model_version"
    ]].to_csv(out_fn, sep="\t", index=False)


if __name__ == "__main__":
    main()


