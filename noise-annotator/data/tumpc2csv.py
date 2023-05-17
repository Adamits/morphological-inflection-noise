import os
import click


def read_src(data_dir, lang):
    def _parse_src_line(line):
        chars, labels = line.rstrip().split("POS=")
        labels, tgt_label = labels.split("TGT_LABEL=")
        labels, src_label = labels.split("SRC_LABEL=")
        pos_label = labels.rstrip()

        return (
            "".join(chars.split()),
            f"{pos_label}_{src_label.rstrip()}",
            f"{pos_label}_{tgt_label.rstrip()}"
        )

    train_path = os.path.join(data_dir, f"{lang}.infl-train.src")
    valid_path = os.path.join(data_dir, f"{lang}.infl-valid.src")

    train_samples = []
    with open(train_path, "r") as file:
        for line in file:
            train_samples.append(_parse_src_line(line))

    valid_samples = []
    with open(valid_path, "r") as file:
        for line in file:
            valid_samples.append(_parse_src_line(line))

    return train_samples + valid_samples
            
    
def read_tgt(data_dir, lang):
    train_path = os.path.join(data_dir, f"{lang}.infl-train.tgt")
    valid_path = os.path.join(data_dir, f"{lang}.infl-valid.tgt")

    train_samples = []
    with open(train_path, "r") as file:
        for line in file:
            train_samples.append("".join(line.rstrip().split()))

    valid_samples = []
    with open(valid_path, "r") as file:
        for line in file:
            valid_samples.append("".join(line.rstrip().split()))

    return train_samples + valid_samples


@click.command()
@click.option("--data_dir", required=True)
@click.option("--lang", required=True)
@click.option("--output_path", required=True)
def main(data_dir, lang, output_path):
    src = read_src(data_dir, lang)
    tgt = read_tgt(data_dir, lang)

    with open(output_path, "w") as out:
        header = "src word,tgt word,src slot,tgt slot"
        print(
            header,
            file=out
        )
        
        for (src_word, src_label, tgt_label), tgt_word in zip(src, tgt):
            
            # Do not include identity inflection
            # if src_label == tgt_label and src_word == tgt_word:
            #     continue
            print(
                ",".join([src_word, tgt_word, src_label, tgt_label]),
                file=out
            )

if __name__ == "__main__":
    main()