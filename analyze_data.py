
import pandas as pd
import editdistance
import seaborn as sns
import matplotlib.pyplot as plt
import os
import requests
import shutil
from pyclts import CLTS
import unidecode
import numpy as np
from scipy.spatial.distance import pdist, squareform

plt.rcParams['savefig.dpi'] = 300

currentdir = os.path.dirname(os.path.realpath(__file__))

CLTS_ARCHIVE_PATH = os.path.join(currentdir, "2.1.0.tar.gz")
CLTS_ARCHIVE_URL = "https://github.com/cldf-clts/clts/archive/refs/tags/v2.1.0.tar.gz"
CLTS_PATH = os.path.join(currentdir, "clts-2.1.0")

OUTPUT_DIR = "output_data"
OUTPUT_DIR_PROTO = os.path.join(OUTPUT_DIR, "proto")
OUTPUT_DIR_MODERN = os.path.join(OUTPUT_DIR, "modern")

pd.set_option('display.max_rows', 100)
img_extension = "png"

######################################## Not used at the moment

def download_if_needed(archive_path, archive_url, file_path, label):
    if not os.path.exists(file_path):
        # Create parent dirs
        #p = pathlib.Path(file_path)
        #p.parent.mkdir(parents=True, exist_ok=True)
        with open(archive_path, 'wb') as f:
            print(f"Downloading {label} from {archive_url}")
            try:
                r = requests.get(archive_url, allow_redirects=True)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                raise SystemExit(e)
            # Write downloaded content to file
            f.write(r.content)
            if archive_path.endswith(".tar.gz"):
                print("Unpacking archive.")
                shutil.unpack_archive(archive_path, currentdir)

def load_clts():
    # Download CLTS
    download_if_needed(CLTS_ARCHIVE_PATH, CLTS_ARCHIVE_URL, CLTS_PATH, "CLTS")


def ipa_to_soundclass(ipa_string):
    asjp_string = []
    for char_ipa in ipa_string:
        char_asjp = clts.bipa.translate(char_ipa, asjp)
        # Only add ASJP char if original char was recognized. Otherwise add original
        if char_asjp == "?":
            asjp_string.append(char_ipa)
        else:
            asjp_string.append(char_asjp)
    # ipa_string_spaced = " ".join(ipa_string)
    # asjp_string_spaced = clts.bipa.translate(ipa_string_spaced, asjp)
    # asjp_string = asjp_string_spaced.replace(" ", "")
    # if "?" in asjp_string:
    #     print(f"{ipa_string} ||| {ipa_string_spaced} ||| {asjp_string_spaced} ||| {asjp_string}")
    return "".join(asjp_string)

##########################################

def normalized_levenshtein(a,b):
    max_len = max(len(a),len(b))
    return editdistance.eval(a,b) /max_len if max_len > 0 else 0

def get_first(x):
    return x[0]


def main():
    # load_clts()
    # clts = CLTS(f"{currentdir}/clts-2.1.0")
    # asjp = clts.soundclass("asjp")

    if not os.path.exists(OUTPUT_DIR_PROTO):
        os.makedirs(OUTPUT_DIR_PROTO)
    if not os.path.exists(OUTPUT_DIR_MODERN):
        os.makedirs(OUTPUT_DIR_MODERN)

    df = pd.read_csv("data/verbal_person-number_indexes_merged.csv")
    # print(df.columns)
    # print(df.groupby("proto_language")["language"].count())

    # Filter out entries without form or protoform (removes languages without protolanguage + possibly more)
    df = df[df['modern_form'].notna()]
    df = df[df['proto_form'].notna()]


    # Find languages which have both protoform and modern form with length 0
    languages_00 = df[(df["modern_length"]==0.0) & (df["proto_length"]==0.0)][["language","proto_language"]]
    df = df[~df["language"].isin(languages_00["language"])]

    # Show number of languages per family
    nunique_family = df.groupby("proto_language")["language"].nunique()
    # print(nunique_family)
    families_above_threshold = nunique_family[nunique_family >= 10]

    ### Analysis length: difference modern form and protoform
    df["proto_diff_length"] = (df["modern_length"] - df["proto_length"])
    # for a in df.groupby("proto_language")
    grouped_length = df.groupby(["person_number", "proto_language"]).mean().sort_values("proto_language")
    #print(grouped.head)
    #grouped.to_csv("output.csv")

    ## Analysis length: difference between modern forms within language family
    print(df.groupby(["proto_language", "person_number"])["modern_length"].aggregate(lambda x: np.mean(pdist(np.array(x)[np.newaxis].T))))
    # Also calculate std
    
    # for i, df in df.groupby(["proto_language", "person_number"]):
    #     print(df[["language","person_number", "modern_length"]])
    #     print(df["modern_length"].aggregate(lambda x: pdist(np.array(x)[np.newaxis].T)).mean())
    #     #break
    return


    ### Analysis distance in forms

    for form_type in ["modern_form", "proto_form"]:
        ## Split alternative forms based on delimiters , and /, and take first
        # TODO: Save alternative forms
        df[f"{form_type}_corr"] = df[form_type].str.split(",|/").apply(get_first)
        
        ## Delete parts in brackets
        # TODO: Create alternative forms based on letters in brackets os(i)
        brackets = df[f"{form_type}_corr"].str.contains("\(.+\)")
        # print(brackets.value_counts())
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("\(.+\)", "", regex=True)

        ## Delete dashes
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("-", "", regex=False)
        ## Delete 2 (from h2)
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("2", "", regex=False)
        ## Delete 0 (empty person marker is just represented by empty string)
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("0", "", regex=False)
        ## Delete ø (empty person marker is just represented by empty string)
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("ø", "", regex=False)
        ## Delete *
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("*", "", regex=False)

        # df[f"{form_type}_corr"] = df[f"{form_type}_corr"].apply(ipa_to_soundclass)
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].apply(unidecode.unidecode)

    df["proto_levenshtein"] = df.apply(lambda x: normalized_levenshtein(x["modern_form_corr"], x["proto_form_corr"]), axis=1)
    # Edit dist, grouped per language family
    grouped_proto_levenshtein = df.groupby(["person_number", "proto_language"]).mean().sort_values("proto_language")


    ### Create all plots 

    sns.violinplot(x="person_number", y="proto_levenshtein", data=df) # hue="proto_language"
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_violin.{img_extension}"))
    plt.clf()
    sns.stripplot(x="person_number", y="proto_levenshtein", data=df)
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_strip.{img_extension}"))
    plt.clf()

    sns.violinplot(x="person_number", y="proto_diff_length", data=df) # hue="proto_language"
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_violin.{img_extension}"))
    plt.clf()
    sns.stripplot(x="person_number", y="proto_diff_length", data=df)
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_strip.{img_extension}"))
    plt.clf()

    for fam, group in df.groupby("proto_language"):
        if fam not in families_above_threshold:
            continue
        sns.violinplot(x="person_number", y="proto_levenshtein", data=group) # hue="proto_language"
        plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_violin_{fam}.{img_extension}"))
        plt.clf()
        sns.stripplot(x="person_number", y="proto_levenshtein", data=group)
        plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_strip_{fam}.{img_extension}"))
        plt.clf()

        sns.violinplot(x="person_number", y="proto_diff_length", data=group) # hue="proto_language"
        plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_violin_{fam}.{img_extension}"))
        plt.clf()
        sns.stripplot(x="person_number", y="proto_diff_length",data=group)
        plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_strip_{fam}.{img_extension}"))
        plt.clf()


    df[["language","modern_form", "modern_form_corr", "proto_form", "proto_form_corr", "modern_length", "proto_length", "proto_diff_length","proto_levenshtein"]].to_csv(os.path.join(OUTPUT_DIR,"metrics.csv"))


if __name__ == "__main__":
    main()
    