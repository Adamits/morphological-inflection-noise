import requests


URL = "https://dictionary.yandex.net/api/v1/dicservice.json/lookup"
API_KEY = "dict.1.1.20220815T201036Z.42721b6b6a348db8.ad78694146c3e6e9151ce74b36d7339983df8a74"


def lookup(text):
    r = requests.get(f"{URL}?key={API_KEY}&lang=ru-en&text={text}")
    print(r.json())
    return r.json()["def"]


def main(csv_path, outpath):
    with open(outpath, "w") as out:
        with open(csv_path, "r") as csv_file:
            next(csv_file)
            for line in csv_file:
                word, src, pos, trans, *_ = line.split(",")
                print(f"Searching for {word}")
                results = lookup(word)
                if len(results) > 0:
                    out.write(",".join([
                        word,
                        results["pos"],
                        "https://dictionary.yandex.net",
                        results["tr"]
                       ])
                    )
                else:
                    out.write(",".join([
                        word,
                        "",
                        "",
                        ""
                       ])
                    )

                out.write("\n")



if __name__ == "__main__":
    c = "/Users/adamwiemerslage/nlp-projects/morphology/noise-annotator/scripts/Russian Filtered Lex Errors - Sheet1.csv"
    o = "Russian\ Yandex\ Filtered\ Lex\ Errors\ -\ Sheet1.csv"
    main(c, o)