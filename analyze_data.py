
import pandas as pd
import editdistance
import seaborn as sns
import matplotlib.pyplot as plt
import os
# from pyclts import CLTS
import unidecode
import numpy as np
from scipy.spatial.distance import pdist, squareform
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from itertools import combinations


# import rpy2.robjects as robjects
# import rpy2.robjects.numpy2ri
# import rpy2.robjects.pandas2ri
# robjects.numpy2ri.activate()
# robjects.pandas2ri.activate()

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

person_markers = ["1sg", "2sg", "3sg", "1pl", "2pl", "3pl"]





######################################## Not used at the moment

# def download_if_needed(archive_path, archive_url, file_path, label):
#     if not os.path.exists(file_path):
#         # Create parent dirs
#         #p = pathlib.Path(file_path)
#         #p.parent.mkdir(parents=True, exist_ok=True)
#         with open(archive_path, 'wb') as f:
#             print(f"Downloading {label} from {archive_url}")
#             try:
#                 r = requests.get(archive_url, allow_redirects=True)
#             except requests.exceptions.RequestException as e:  # This is the correct syntax
#                 raise SystemExit(e)
#             # Write downloaded content to file
#             f.write(r.content)
#             if archive_path.endswith(".tar.gz"):
#                 print("Unpacking archive.")
#                 shutil.unpack_archive(archive_path, currentdir)

# def load_clts():
#     # Download CLTS
#     download_if_needed(CLTS_ARCHIVE_PATH, CLTS_ARCHIVE_URL, CLTS_PATH, "CLTS")


# def ipa_to_soundclass(ipa_string):
#     asjp_string = []
#     for char_ipa in ipa_string:
#         char_asjp = clts.bipa.translate(char_ipa, asjp)
#         # Only add ASJP char if original char was recognized. Otherwise add original
#         if char_asjp == "?":
#             asjp_string.append(char_ipa)
#         else:
#             asjp_string.append(char_asjp)
#     # ipa_string_spaced = " ".join(ipa_string)
#     # asjp_string_spaced = clts.bipa.translate(ipa_string_spaced, asjp)
#     # asjp_string = asjp_string_spaced.replace(" ", "")
#     # if "?" in asjp_string:
#     #     print(f"{ipa_string} ||| {ipa_string_spaced} ||| {asjp_string_spaced} ||| {asjp_string}")
#     return "".join(asjp_string)

##########################################

def normalized_levenshtein(a,b):
    max_len = max(len(a),len(b))
    return editdistance.eval(a,b) # /max_len if max_len > 0 else 0

def get_first(x):
    return x[0]

def df_stats(df, label):
    n_entries = len(df)
    n_languages = df["language"].nunique()
    n_proto_languages = df["proto_language"].nunique()
    nunique_family = df.groupby("proto_language")["language"].nunique()
    print(f"{label}: entries: {n_entries}, languages: {n_languages}, proto_languages: {n_proto_languages}, n_unique_family: {nunique_family}.")

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
    df_stats(df, "original")
    # Filter out entries without form or protoform (removes languages without protolanguage + possibly more)
    df = df[df['modern_form'].notna()]
    df_stats(df, "after removing modern NA")
    df = df[df['proto_form'].notna()]
    #languages_one_protoform_na = df[df["proto_form"].isna()][["language"]]
    #df = df[~df["language"].isin(languages_one_protoform_na["language"])]
    df_stats(df, "after removing proto NA")


    # Find languages which have both protoform and modern form with length 0
    #languages_00 = df[(df["modern_length"]==0.0) & (df["proto_length"]==0.0)][["language","proto_language"]]
    languages_0 = df[df["proto_length"]==0.0][["language"]]
    df = df[~df["language"].isin(languages_0["language"])] # remove all languages where protolanguage is 0
    # Replace by languages_00 to remove only languages where proto and modern form are 0
    df_stats(df, "after removing languages where one protoform has length 0")

    # Show number of languages per family
    nunique_family = df.groupby("proto_language")["language"].nunique()
    # print(nunique_family)
    families_above_threshold = nunique_family[nunique_family >= 10]

    ### Analysis length: difference modern form and protoform
    df["proto_diff_length"] = (df["modern_length"] - df["proto_length"])
    # for a in df.groupby("proto_language")
    # grouped_length = df.groupby(["proto_language", "person_number"]).mean().sort_values("proto_language")
    #print(grouped.head)
    #grouped.to_csv("output.csv")
    
    ### Analysis levenshtein distance in forms

    for form_type in ["modern_form", "proto_form"]:
        ## Split alternative forms based on delimiters , and /, and take first
        # TODO: Save alternative forms
        df[f"{form_type}_corr"] = df[form_type].str.split(",|/").apply(get_first)
        
        ## Delete parts in brackets
        # TODO: Create alternative forms based on letters in brackets os(i)
        #brackets = df[f"{form_type}_corr"].str.contains("\(.+\)")
        # print(brackets.value_counts())
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("\(.+\)", "", regex=True)

        ## Delete dashes
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("-", "", regex=False)
        ## Delete ... (non-concatenative morphology)
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("...", "", regex=False)
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("…", "", regex=False)
        ## Delete 2 (from h2)
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("2", "", regex=False)
        ## Delete 0 (empty person marker is just represented by empty string)
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("0", "", regex=False)
        ## Delete ø (empty person marker is just represented by empty string)
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("ø", "", regex=False)
        ## Delete *
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("*", "", regex=False)
        ## Delete ´ ' # (segments which are not counted in precalculated length)
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace("[´`'#]", "", regex=True)
        ## Delete : (lengthening vowel but no sound on its own)
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].str.replace(":", "", regex=False)
        

        # df[f"{form_type}_corr"] = df[f"{form_type}_corr"].apply(ipa_to_soundclass)
        df[f"{form_type}_corr"] = df[f"{form_type}_corr"].apply(unidecode.unidecode)
    
    df.to_csv("test.csv")

    df["proto_levenshtein"] = df.apply(lambda x: normalized_levenshtein(x["modern_form_corr"], x["proto_form_corr"]), axis=1)
    # Edit dist, grouped per language family
    # grouped_proto_levenshtein = df.groupby(["person_number", "proto_language"]).mean().sort_values("proto_language")

    ### Do statistical analyses
    persons_numbers = [(p,n) for n in ["sg","pl"] for p in ["first","second","third"] ]
    pn_matrix = pd.DataFrame(persons_numbers, columns=["person","number"])
    
    # Poisson mixed model: https://www.statsmodels.org/stable/generated/statsmodels.genmod.bayes_mixed_glm.PoissonBayesMixedGLM.html#statsmodels.genmod.bayes_mixed_glm.PoissonBayesMixedGLM
    # Easier, GLM with family argument: https://www.kaggle.com/code/hongpeiyi/poisson-regression-with-statsmodels

    print("Regression proto diff length")
    mixedlm_proto_diff_length = smf.mixedlm("proto_diff_length ~ person*C(number, Treatment('sg'))", groups=df["clade3"], data = df).fit()
    #mixedlm_proto_diff_length = smf.mixedlm("proto_diff_length ~ person_number", groups=df["proto_language"], data = df).fit()
    print(mixedlm_proto_diff_length.summary())
    pn_matrix["predictions_mixedlm_proto_diff_length"] = mixedlm_proto_diff_length.predict(pn_matrix)
    # print(mixedlm_proto_diff_length.t_test(matrix))
    # TODO: How to get predictions with their own standard deviations?
    # https://tedboy.github.io/statsmodels_doc/generated/statsmodels.sandbox.regression.predstd.wls_prediction_std.html#statsmodels.sandbox.regression.predstd.wls_prediction_std
    # Use t-test? https://stats.stackexchange.com/questions/578398/getting-confidence-interval-for-prediction-from-statsmodel-robust-linear-model

    print("Regression proto Levenshtein")
    mixedlm_proto_levenshtein = smf.mixedlm("proto_levenshtein ~ person*C(number, Treatment('sg'))", groups=df["clade3"], data = df).fit()
    print(mixedlm_proto_levenshtein.summary())
    pn_matrix["predictions_mixedlm_proto_levenshtein"] = mixedlm_proto_levenshtein.predict(pn_matrix)

    print(pn_matrix)

    print("Regression proto diff length, joint person-number")
    mixedlm_proto_diff_length_pn_joint = smf.mixedlm("proto_diff_length ~ person_number + 0 ", groups=df["clade3"], data = df).fit()
    print(mixedlm_proto_diff_length_pn_joint.params)

    print("Regression proto Levenshtein, joint person-number")
    mixedlm_proto_levenshtein_pn_joint = smf.mixedlm("proto_levenshtein ~ person_number + 0", groups=df["clade3"], data = df).fit()
    print(mixedlm_proto_levenshtein_pn_joint.params)

    ## Statistical analyses in R
    # with robjects.local_context() as lc:
    #     lc['df'] = df
    #     lc['pnMatrix'] = pn_matrix

    #     robjects.r('''
    #             library(tidyverse)
    #             library(lme4)
    #             df <- mutate(df,
    #                         number = relevel(factor(number), ref = 'sg'))
    #             modelDiffLength <- lmer(proto_diff_length ~ person*number + (1|clade3), df)
    #             fixefsDiffLength <- fixef(modelDiffLength)
    #             predictionsDiffLength <- predict(modelDiffLength, pnMatrix, re.form=NA)
    #             modelProtoLev <- lmer(proto_levenshtein ~ person*number + (1|clade3), df)
    #             predictionsProtoLev <- predict(modelProtoLev, pnMatrix, re.form=NA)
    #             fixefsProtoLev <- fixef(modelProtoLev)
    #             ''')
        
    #     print(lc['modelDiffLength'])
    #     print(lc['predictionsDiffLength'])
    #     print(lc['modelProtoLev'])
    #     print(lc['predictionsProtoLev'])
    return

    # print("Regression modern diff length")
    # regression_modern_diff_length = ols("modern_diff_length ~ person_number + 0", data = df_modern_pairwise).fit()
    # print(regression_modern_diff_length.params)

    # print("Regression modern Levenshtein")
    # regression_modern_levenshtein = ols("modern_levenshtein ~ person_number + 0", data = df_modern_pairwise).fit()
    # print(regression_modern_levenshtein.params)


    ## Analysis length: pairwise difference between modern forms within language family
    # With full groupby aggregate, we can only get one value (mean) per language family. (or maybe one value per language)
    # We want every comparison as separate data point
    # print(df.groupby(["proto_language", "person_number"])["modern_length"].aggregate(lambda x: pdist(np.array(x)[np.newaxis].T)))
    modern_pairwise_dfs = []
    for (pl,pn), group in df.groupby(["proto_language", "person_number"]):
        # print(group[["proto_language", "person_number", "modern_form", "modern_length"]])
        modern_diff_length = group["modern_length"].aggregate(lambda x: pdist(np.array(x)[np.newaxis].T))
        modern_levenshtein = group["modern_form_corr"].aggregate(lambda x: [normalized_levenshtein(a,b) for a,b in combinations(np.array(x),2)])
        # print(modern_diff_length)
        # print(modern_levenshtein)
        d = pd.DataFrame()
        d["modern_diff_length"] = modern_diff_length
        d["modern_levenshtein"] = modern_levenshtein
        d["proto_language"] = pl
        d["person_number"] = pn
        #print(d)
        modern_pairwise_dfs.append(d)
    
    df_modern_pairwise = pd.concat(modern_pairwise_dfs)

    ### Create all plots 
    sns.violinplot(x="person_number", y="proto_diff_length", data=df) # hue="proto_language"
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_violin.{img_extension}"))
    plt.clf()
    sns.stripplot(x="person_number", y="proto_diff_length", data=df)
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_strip.{img_extension}"))
    plt.clf()
    sns.boxplot(x="person_number", y="proto_diff_length", data=df)
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_box.{img_extension}"))
    plt.clf()
    sns.boxplot(x="person", hue="number", y="proto_diff_length", data=df)
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_box_person_number.{img_extension}"))
    plt.clf()

    sns.violinplot(x="person_number", y="proto_levenshtein", data=df) # hue="proto_language"
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_violin.{img_extension}"))
    plt.clf()
    sns.stripplot(x="person_number", y="proto_levenshtein", data=df)
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_strip.{img_extension}"))
    plt.clf()
    sns.boxplot(x="person_number", y="proto_levenshtein", data=df)
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_box.{img_extension}"))
    plt.clf()
    sns.boxplot(x="person", hue="number", y="proto_levenshtein", data=df)
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_box_person_number.{img_extension}"))
    plt.clf()

    ## Modern pairwise length difference
    sns.violinplot(x="person_number", y="modern_diff_length", data=df_modern_pairwise, order=person_markers)
    plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_diff_length_violin.{img_extension}"))
    plt.clf()
    sns.stripplot(x="person_number", y="modern_diff_length", data=df_modern_pairwise, order=person_markers)
    plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_diff_length_strip.{img_extension}"))
    plt.clf()

    ## Modern pairwise Levenshtein
    sns.violinplot(x="person_number", y="modern_levenshtein", data=df_modern_pairwise, order=person_markers)
    plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_levenshtein_violin.{img_extension}"))
    plt.clf()
    sns.stripplot(x="person_number", y="modern_levenshtein", data=df_modern_pairwise, order=person_markers)
    plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_levenshtein_strip.{img_extension}"))
    plt.clf()

    # Protolanguage, per family
    for fam, group in df.groupby("proto_language"):
        if fam not in families_above_threshold:
            continue

        sns.violinplot(x="person_number", y="proto_diff_length", data=group) # hue="proto_language"
        plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_violin_{fam}.{img_extension}"))
        plt.clf()
        sns.stripplot(x="person_number", y="proto_diff_length",data=group)
        plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_strip_{fam}.{img_extension}"))
        plt.clf()

        sns.violinplot(x="person_number", y="proto_levenshtein", data=group) # hue="proto_language"
        plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_violin_{fam}.{img_extension}"))
        plt.clf()
        sns.stripplot(x="person_number", y="proto_levenshtein", data=group)
        plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_strip_{fam}.{img_extension}"))
        plt.clf()
    
    ## Modern pairwise, per family
    for fam, group in df_modern_pairwise.groupby("proto_language"):
        if fam not in families_above_threshold:
            continue
        # Modern pairwise length difference
        sns.violinplot(x="person_number", y="modern_diff_length", data=group, order=person_markers)
        plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_diff_length_violin_{fam}.{img_extension}"))
        plt.clf()
        sns.stripplot(x="person_number", y="modern_diff_length", data=group, order=person_markers)
        plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_diff_length_strip_{fam}.{img_extension}"))
        plt.clf()

        sns.violinplot(x="person_number", y="modern_levenshtein", data=group, order=person_markers) # hue="proto_language"
        plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_levenshtein_violin_{fam}.{img_extension}"))
        plt.clf()
        sns.stripplot(x="person_number", y="modern_levenshtein", data=group, order=person_markers)
        plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_levenshtein_strip_{fam}.{img_extension}"))
        plt.clf()



    df[["language","modern_form", "modern_form_corr", "proto_form", "proto_form_corr", "modern_length", "proto_length", "proto_diff_length","proto_levenshtein"]].to_csv(os.path.join(OUTPUT_DIR,"metrics.csv"))


if __name__ == "__main__":
    main()
    