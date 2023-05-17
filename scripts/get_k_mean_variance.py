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

def make_mc_output(base_path, arch, lang):
    outputs = []
    arch_path = os.path.join(base_path, arch, lang)
    eval_filename = os.path.join(arch_path, "f.stats")

    last_aggregate = False
    if not os.path.isfile(eval_filename):
        print(f"WARNING: No such file {eval_filename}")
        print("Skipping...")
        return 0, 0

    with open(eval_filename, "r") as f:
        for line in f:
            if line.startswith("DEV ACCURACY (internal evaluation)"):
                acc = line.rstrip().split("= ")[1]
                return float(acc), 0


@click.command()
@click.option("--results_dir", required=True)
@click.option("--lang", required=True)
def main(results_dir, lang):
    arches = [
        arch for arch in os.listdir(os.path.join(results_dir, "1"))
    ]
    arch_data = {
        arch: {"Acc": [], "Epoch": [], "K": []} for arch in set(arches)
    }

    for k in os.listdir(results_dir):
        base_path = os.path.join(results_dir, k)
        for arch in os.listdir(base_path):
            if arch.startswith("M_C"):
                acc, epoch = make_mc_output(base_path, arch, lang)
                arch_data[arch]["Acc"].append(round(acc * 100, 2))
                arch_data[arch]["Epoch"].append(int(epoch))
                arch_data[arch]["K"].append(k)
                continue

            arch_path = os.path.join(base_path, arch)
            lang_path = os.path.join(arch_path, lang)
            versions = os.listdir(lang_path)
            max_version_num = max([int(v.split("_")[-1]) for v in versions])
            file_name = os.path.join(lang_path, f"version_{max_version_num}", "metrics.csv")

            if not os.path.isfile(file_name):
                print(f"WARNING: No such file {file_name}")
                print("Skipping...")
                continue

            df = pd.read_csv(file_name)
            max_val_acc_idx = df["val_accuracy"].idxmax()
            arch_data[arch]["Acc"].append(round(df.iloc[max_val_acc_idx].val_accuracy * 100, 2))
            arch_data[arch]["Epoch"].append(int(df.iloc[max_val_acc_idx].epoch))
            arch_data[arch]["K"].append(k)

    for arch, data in arch_data.items():
        print(arch)
        print(data["Epoch"])
        accs = data["Acc"]
        print(accs)
        print("\nMean: ", np.mean(accs), "Std Dev: ", np.std(accs), "Variance: ", np.var(accs))

if __name__ == "__main__":
    main()


