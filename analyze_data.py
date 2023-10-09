
# analyze_data.py: Standalone script to perform statistical analysis of length change in data of Serzant & Moroz (2022).
# Input data expected to be in: data/verbal_person-number_indexes_merged.csv
# See README for more information on installation.

import pandas as pd
import editdistance
import seaborn as sns
import matplotlib.pyplot as plt
import os
import unidecode
import numpy as np
from itertools import combinations

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects.pandas2ri
robjects.numpy2ri.activate()
robjects.pandas2ri.activate()

plt.rcParams['savefig.dpi'] = 300

currentdir = os.path.dirname(os.path.realpath(__file__))

# CLTS_ARCHIVE_PATH = os.path.join(currentdir, "2.1.0.tar.gz")
# CLTS_ARCHIVE_URL = "https://github.com/cldf-clts/clts/archive/refs/tags/v2.1.0.tar.gz"
# CLTS_PATH = os.path.join(currentdir, "clts-2.1.0")

OUTPUT_DIR = "output_data"
OUTPUT_DIR_PROTO = os.path.join(OUTPUT_DIR, "proto")
OUTPUT_DIR_MODERN = os.path.join(OUTPUT_DIR, "modern")

# User-settable params:
EXCLUDE_LANGUAGES_PROTO_0 = False # Exclude languages (and thus whole families) where one of the protoforms is zero
NORMALISATION = "max"
excl_proto0_label = "_exclproto0" if EXCLUDE_LANGUAGES_PROTO_0 else ""
norm_label = f"_{NORMALISATION}"
NORM_STRING_TITLE = "normalised " if NORMALISATION is not "none" else "" # This assumes always 'max' normalisation, other types get the same label

pd.set_option('display.max_rows', 100)
img_extension_pyplots = "png"

person_markers = ["1sg", "2sg", "3sg", "1pl", "2pl", "3pl"]



def normalised_levenshtein(modern,proto, norm):
    raw_dist = editdistance.eval(modern, proto)
    if norm == "mean":
        norm_len = np.mean([len(modern),len(proto)])
    elif norm == "max":
        norm_len = max(len(modern), len(proto))
    elif norm=="sqrt":
        norm_len = np.sqrt(np.mean([len(modern),len(proto)]))
    elif norm=="none":
        norm_len = 1
    else:
        raise ValueError("norm should be one of 'mean' or 'max'.")
    return raw_dist / norm_len if norm_len > 0 else 0


##########################################


def get_first(x):
    return x[0]

def stats_df(df, label):
    n_entries = len(df)
    n_languages = df["language"].nunique()
    n_proto_languages = df["proto_language"].nunique()
    # nunique_family = df.groupby("proto_language")["language"].nunique()
    print(f"{label}: entries: {n_entries}, languages: {n_languages}, proto_languages: {n_proto_languages}")

def main():

    if not os.path.exists(OUTPUT_DIR_PROTO):
        os.makedirs(OUTPUT_DIR_PROTO)
    if not os.path.exists(OUTPUT_DIR_MODERN):
        os.makedirs(OUTPUT_DIR_MODERN)

    df = pd.read_csv("data/verbal_person-number_indexes_merged.csv")


    # Reporting: Create an excerpt of Serzant & Moroz (2022) data (for SI)
    df[["language", "proto_language", "person_number", "person", "number", "modern_form", "proto_form", "clade3"]].head(18).to_latex(os.path.join(OUTPUT_DIR,"excerpt_serzantmoroz2022.tex"))


    df = df.drop(columns=["source", "comment", "proto_source", "proto_comments", "changed_GM"])
    stats_df(df, "original")
    # Filter out entries without form or protoform (removes languages without protolanguage + possibly more)
    df = df[df['modern_form'].notna()]
    stats_df(df, "after removing modern NA")
    df = df[df['proto_form'].notna()]
    #languages_one_protoform_na = df[df["proto_form"].isna()][["language"]]
    #df = df[~df["language"].isin(languages_one_protoform_na["language"])]
    stats_df(df, "after removing proto NA")

    # Reporting: Creating tables with zero forms, to aid discussion in paper
    proto_lengths = df.groupby(["proto_language","person_number"]).first()["proto_length"]
    proto_lengths.to_csv(os.path.join(OUTPUT_DIR,"proto_lengths_fam.csv"))
    proto_lengths_zero = proto_lengths[proto_lengths == 0.0]
    # proto_lengths_zero = proto_lengths[proto_lengths["proto_length"] == 0.0]
    proto_lengths_zero.to_csv(os.path.join(OUTPUT_DIR,"proto_lengths_fam_zero.csv"))
    modern_reflexes_proto_lengths_zero = pd.merge(df, proto_lengths_zero, on=["proto_language", "person_number"])
    modern_reflexes_proto_lengths_zero.to_csv(os.path.join(OUTPUT_DIR,"modern_reflexes_proto_zero.csv"))


    # Find languages which have both protoform and modern form with length 0
    if EXCLUDE_LANGUAGES_PROTO_0:
        #languages_00 = df[(df["modern_length"]==0.0) & (df["proto_length"]==0.0)][["language","proto_language"]]
        languages_proto0 = df[df["proto_length"]==0.0][["language"]]
        df = df[~df["language"].isin(languages_proto0["language"])] # remove all languages where protolanguage is 0
        stats_df(df, "after removing languages where one protoform has length 0")

    # Show number of languages per family
    nunique_family = df.groupby("proto_language")["language"].nunique()
    # print(nunique_family)

    ### Analysis length: difference modern form and protoform
    df["proto_diff_length"] = (df["modern_length"] - df["proto_length"])
    
    ### Analysis levenshtein distance in forms

    for form_type in ["modern_form", "proto_form"]:
        ## Split alternative forms based on delimiters , and /, and take first
        df[f"{form_type}_corr"] = df[form_type].str.split(",|/").apply(get_first)
        
        ## Delete parts in brackets
        # Possible future: Create alternative forms based on letters in brackets os(i)
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

    df["proto_levenshtein"] = df.apply(lambda x: normalised_levenshtein(x["modern_form_corr"], x["proto_form_corr"], NORMALISATION), axis=1)

    df["person_merged"] = df["person"].apply(lambda p: "third" if p=="third" else "firstsecond")

    ## Reporting: calculate proportion of forms with Levenshtein distance 0 that also have protoform 0
    print("Distribution of persons in dataset")
    distr_persons_dataset = df["person_number"].value_counts()
    distr_persons_dataset.to_latex(os.path.join(OUTPUT_DIR,"distr_persons_dataset.tex"))
    print(distr_persons_dataset)

    print("Distribution of proto lengths in all data:")
    distr_proto_dataset = df["proto_length"].value_counts()
    distr_proto_dataset.to_latex(os.path.join(OUTPUT_DIR,"distr_proto_dataset.tex"))
    print(distr_proto_dataset)

    print("Distribution of persons where protoform is empty")
    distr_persons_proto0 = df[df["proto_length"]==0.0]["person_number"].value_counts()
    distr_persons_proto0.to_latex(os.path.join(OUTPUT_DIR,"distr_persons_proto0.tex"))
    print(distr_persons_proto0)

    df_levenshtein0 = df[df["proto_levenshtein"]==0.0]
    print(f"Entries with Levenshtein 0: {len(df_levenshtein0)} (total entries: {len(df)})")
    print("Distribution of proto lengths in entries with Levenshtein 0:")
    distr_proto_lev0 = df_levenshtein0["proto_length"].value_counts()
    distr_proto_lev0.to_latex(os.path.join(OUTPUT_DIR,"distr_proto_lev0.tex"))
    print(distr_proto_lev0)

    print("Distribution of persons, of entries with Levenshtein 0, and where protoform is empty")
    distr_persons_lev0_proto0 = df_levenshtein0[df_levenshtein0["proto_length"]==0.0]["person_number"].value_counts()
    print(distr_persons_lev0_proto0)
    distr_persons_lev0_proto0.to_latex(os.path.join(OUTPUT_DIR,"distr_persons_lev0_proto0.tex"))
    

    ## Statistical analyses in R
    with robjects.local_context() as lc:
        lc['df'] = df


        robjects.r(f'''
                library(tidyverse)
                library(lme4)
                library(ggeffects)
                library(afex)
                df <- mutate(df,
                            number = relevel(factor(number), ref = 'sg'))

                # modelProtoDiffLength <- lmer(proto_diff_length ~ person*number + (1|clade3), data=df)
                # modelProtoDiffLengthSum <- summary(modelProtoDiffLength)
                # predictionsProtoDiffLength <- ggpredict(model=modelProtoDiffLength, terms=c('person','number'))
                # plot(predictionsProtoDiffLength)+
                # ggtitle("Mixed model difference proto and modern length")+
                # labs(y = "proto length - modern length")
                # ggsave("{OUTPUT_DIR_PROTO}/predictions_proto_diff_length{excl_proto0_label}.png", bg = "white")
                # ggsave("{OUTPUT_DIR_PROTO}/predictions_proto_diff_length{excl_proto0_label}.pdf", bg = "white")

                # modelProtoDiffLengthMerged <- lmer(proto_diff_length ~ person_merged*number + (1|clade3), data=df)
                # modelProtoDiffLengthMergedSum <- summary(modelProtoDiffLengthMerged)
                # predictionsProtoDiffLengthMerged <- ggpredict(model=modelProtoDiffLengthMerged, terms=c('person_merged','number'))
                # plot(predictionsProtoDiffLengthMerged)+
                # ggtitle("Mixed model difference proto and modern length merged")+
                # labs(y = "proto length - modern length")
                # ggsave("{OUTPUT_DIR_PROTO}/predictions_proto_diff_length merged{excl_proto0_label}.png", bg = "white")
                # ggsave("{OUTPUT_DIR_PROTO}/predictions_proto_diff_length merged{excl_proto0_label}.pdf", bg = "white")

                # # ANOVA test: diff length
                # modelProtoDiffLengthMergedML <- lmer(proto_diff_length ~ person_merged*number + (1|clade3), data=df, REML=FALSE)
                # modelProtoDiffLengthMergedNoPerson <- lmer(proto_diff_length ~ number + (1|clade3), data=df, REML=FALSE)
                # anovaDiffLengthMerged <- anova(modelProtoDiffLengthMergedNoPerson, modelProtoDiffLengthMergedML, test = 'Chisq')

                modelProtoLev <- lmer(proto_levenshtein ~ person*number + (1|clade3), data=df)
                modelProtoLevSum <- summary(modelProtoLev)
                predictionsProtoLev <- ggpredict(model=modelProtoLev, terms=c('person','number'))
                plot(predictionsProtoLev)+
                ggtitle("Mixed model {NORM_STRING_TITLE}Levenshtein distance proto and modern length")+
                labs(y = "Levenshtein distance")
                ggsave("{OUTPUT_DIR_PROTO}/predictions_proto_levenshtein{excl_proto0_label}{norm_label}.png", bg = "white")
                ggsave("{OUTPUT_DIR_PROTO}/predictions_proto_levenshtein{excl_proto0_label}{norm_label}.pdf", bg = "white")

                # ANOVA test
                # modelProtoLevML <- lmer(proto_levenshtein ~ person*number + (1|clade3), data=df, REML=FALSE)
                # modelProtoLevMLSum <- summary(modelProtoLevML)
                # modelProtoLevNoPerson <- lmer(proto_levenshtein ~ number + (1|clade3), data=df, REML=FALSE)
                # anovaLev <- anova(modelProtoLevNoPerson, modelProtoLevML, test = 'Chisq')
                anovaLevAfex <- mixed(proto_levenshtein ~ person*number + (1|clade3), data=df, method='LRT')

                # modelProtoLevMerged <- lmer(proto_levenshtein ~ person_merged*number + (1|clade3), data=df)
                # modelProtoLevMergedSum <- summary(modelProtoLevMerged)
                # predictionsProtoLevMerged <- ggpredict(model=modelProtoLevMerged, terms=c('person_merged','number'))
                # plot(predictionsProtoLevMerged)+
                # ggtitle("Mixed model Levenshtein distance proto and modern length merged")+
                # labs(y = "Levenshtein distance")
                # ggsave("{OUTPUT_DIR_PROTO}/predictions_proto_levenshtein_merged{excl_proto0_label}{norm_label}.png", bg = "white")
                # ggsave("{OUTPUT_DIR_PROTO}/predictions_proto_levenshtein_merged{excl_proto0_label}{norm_label}.pdf", bg = "white")
                # ANOVA test: proto lev merged
                # modelProtoLevMergedML <- lmer(proto_levenshtein ~ person_merged*number + (1|clade3), data=df, REML=FALSE)
                # modelProtoLevMergedNoPerson <- lmer(proto_levenshtein ~ number + (1|clade3), data=df, REML=FALSE)
                # anovaLevMerged <- anova(modelProtoLevMergedNoPerson, modelProtoLevMergedML, test = 'Chisq')
                ''')

        print(" - Proto Levenshtein")
        print(lc['modelProtoLevSum'])
        print(lc['predictionsProtoLev'])


        print(" - Anova afex")
        print(lc['anovaLevAfex'])


    ## Analysis length: pairwise difference between modern forms within language family
    # With full groupby aggregate, we can only get one value (mean) per language family. (or maybe one value per language)
    # # We want every comparison as separate data point
    # # print(df.groupby(["proto_language", "person_number"])["modern_length"].aggregate(lambda x: pdist(np.array(x)[np.newaxis].T)))
    # modern_pairwise_dfs = []
    # for (pl,pn), group in df.groupby(["proto_language", "person_number"]):
    #     # print(group[["proto_language", "person_number", "modern_form", "modern_length"]])
    #     modern_diff_length = group["modern_length"].aggregate(lambda x: pdist(np.array(x)[np.newaxis].T))
    #     modern_levenshtein = group["modern_form_corr"].aggregate(lambda x: [normalised_levenshtein(a,b, NORMALISATION) for a,b in combinations(np.array(x),2)])
    #     # print(modern_diff_length)
    #     # print(modern_levenshtein)
    #     d = pd.DataFrame()
    #     d["modern_diff_length"] = modern_diff_length
    #     d["modern_levenshtein"] = modern_levenshtein
    #     d["proto_language"] = pl
    #     d["person_number"] = pn
    #     #print(d)
    #     modern_pairwise_dfs.append(d)
    
    # df_modern_pairwise = pd.concat(modern_pairwise_dfs)

    ### Create all plots 
    # sns.violinplot(x="person_number", y="proto_diff_length", data=df) # hue="proto_language"
    # plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_violin{excl_proto0_label}.{img_extension_pyplots}"))
    # plt.clf()
    # sns.stripplot(x="person_number", y="proto_diff_length", data=df)
    # plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_strip{excl_proto0_label}.{img_extension_pyplots}"))
    # plt.clf()

    sns.violinplot(x="person_number", y="proto_levenshtein", data=df) # hue="proto_language"
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_violin{excl_proto0_label}{norm_label}.{img_extension_pyplots}"))
    plt.clf()
    sns.stripplot(x="person_number", y="proto_levenshtein", data=df)
    plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_strip{excl_proto0_label}{norm_label}.{img_extension_pyplots}"))
    plt.clf()
    # sns.boxplot(x="person_number", y="proto_levenshtein", data=df)
    # plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_box{excl_proto0_label}{norm_label}.{img_extension_pyplots}"))
    # plt.clf()
    # sns.boxplot(x="person", hue="number", y="proto_levenshtein", data=df)
    # plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_box_person_number{excl_proto0_label}{norm_label}.{img_extension_pyplots}"))
    # plt.clf()

    # ## Modern pairwise length difference
    # sns.violinplot(x="person_number", y="modern_diff_length", data=df_modern_pairwise, order=person_markers)
    # plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_diff_length_violin{excl_proto0_label}.{img_extension_pyplots}"))
    # plt.clf()
    # sns.stripplot(x="person_number", y="modern_diff_length", data=df_modern_pairwise, order=person_markers)
    # plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_diff_length_strip{excl_proto0_label}.{img_extension_pyplots}"))
    # plt.clf()

    # ## Modern pairwise Levenshtein
    # sns.violinplot(x="person_number", y="modern_levenshtein", data=df_modern_pairwise, order=person_markers)
    # plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_levenshtein_violin{excl_proto0_label}{norm_label}.{img_extension_pyplots}"))
    # plt.clf()
    # sns.stripplot(x="person_number", y="modern_levenshtein", data=df_modern_pairwise, order=person_markers)
    # plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_levenshtein_strip{excl_proto0_label}{norm_label}.{img_extension_pyplots}"))
    # plt.clf()

    # # Protolanguage, per family
    # for fam, group in df.groupby("proto_language"):
    #     if fam not in families_above_threshold:
    #         continue

    #     sns.violinplot(x="person_number", y="proto_diff_length", data=group) # hue="proto_language"
    #     plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_violin_{fam}{excl_proto0_label}.{img_extension_pyplots}"))
    #     plt.clf()
    #     sns.stripplot(x="person_number", y="proto_diff_length",data=group)
    #     plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_diff_length_strip_{fam}{excl_proto0_label}.{img_extension_pyplots}"))
    #     plt.clf()

    #     sns.violinplot(x="person_number", y="proto_levenshtein", data=group) # hue="proto_language"
    #     plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_violin_{fam}{excl_proto0_label}{norm_label}.{img_extension_pyplots}"))
    #     plt.clf()
    #     sns.stripplot(x="person_number", y="proto_levenshtein", data=group)
    #     plt.savefig(os.path.join(OUTPUT_DIR_PROTO,f"proto_levenshtein_strip_{fam}{excl_proto0_label}{norm_label}.{img_extension_pyplots}"))
    #     plt.clf()
    
    # ## Modern pairwise, per family
    # for fam, group in df_modern_pairwise.groupby("proto_language"):
    #     if fam not in families_above_threshold:
    #         continue
    #     # Modern pairwise length difference
    #     sns.violinplot(x="person_number", y="modern_diff_length", data=group, order=person_markers)
    #     plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_diff_length_violin_{fam}{excl_proto0_label}.{img_extension_pyplots}"))
    #     plt.clf()
    #     sns.stripplot(x="person_number", y="modern_diff_length", data=group, order=person_markers)
    #     plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_diff_length_strip_{fam}{excl_proto0_label}.{img_extension_pyplots}"))
    #     plt.clf()

    #     sns.violinplot(x="person_number", y="modern_levenshtein", data=group, order=person_markers) # hue="proto_language"
    #     plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_levenshtein_violin_{fam}{excl_proto0_label}{norm_label}.{img_extension_pyplots}"))
    #     plt.clf()
    #     sns.stripplot(x="person_number", y="modern_levenshtein", data=group, order=person_markers)
    #     plt.savefig(os.path.join(OUTPUT_DIR_MODERN,f"modern_levenshtein_strip_{fam}{excl_proto0_label}{norm_label}.{img_extension_pyplots}"))
    #     plt.clf()



    #df[["language","modern_form", "modern_form_corr", "proto_form", "proto_form_corr", "modern_length", "proto_length", "proto_diff_length","proto_levenshtein"]].to_csv(os.path.join(OUTPUT_DIR,"metrics.csv"))


if __name__ == "__main__":
    main()
    