
# analyze_cgn.py: Standalone script to count persons in Corpus Gesproken Nederlands (CGN, corpus of spoken Dutch)
# Input data expected to be in folder CGN_DIR. By default this is 'data/CGNAnn', with within that the 'Data' folder that contains the whole directory structure.
# See README for more information on installation.

import pandas as pd
import os
from collections import defaultdict
import itertools
import re

CGN_DIR = os.path.join("data", "CGNAnn", "Data", "data", "annot", "text", "syn")

COMPONENTS_FACE_TO_FACE = ["a"]
COMPONENTS_INTERACTION = ["a", "b", "c", "d", "e", "f", "g"]
# COMPONENTS_ALL = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]

def pos_exact_to_person(pos_exact):
    if re.search("VNW\(pers,pron,.*1.*,ev", pos_exact):
        return "1sg"
    elif re.search("VNW\(pers,pron,.*2.*,ev", pos_exact):
        return "2sg"
    elif re.search("VNW\(pers,pron,.*3.*,ev", pos_exact):
        return "3sg"
    elif re.search("N\(.*ev", pos_exact):
        return "3sg-noun"
    elif re.search("WW\(", pos_exact):
        return "3sg-verb"
    elif re.search("VNW\(aanw,pron.*ev", pos_exact):
        return "3sg-demonstr-pronoun"
    elif re.search("VNW\(aanw,pron", pos_exact): # Overlaps with previous regex, depends on evaluation order
        return "3sg/pl-demonstr-pronoun"
    elif re.search("VNW\(aanw,adv-pron", pos_exact):
        return "3sg/pl-demonstr-advpron"
    elif re.search("ADJ\(nom,basis,.*zonder-n.*", pos_exact):
        return "3sg-adj"
    elif re.search("VNW\(pers,pron,.*1.*,mv", pos_exact):
        return "1pl"
    elif re.search("VNW\(pers,pron,.*2.*,mv", pos_exact):
        return "2pl"
    elif re.search("VNW\(pers,pron,.*3.*,mv", pos_exact):
        return "3pl"
    elif re.search("N\(.*mv", pos_exact):
        return "3pl-noun"
    elif re.search("ADJ\(nom,basis,.*mv-n.*", pos_exact):
        return "3pl-adj"
    elif re.search("VNW\(onbep.*mv-n", pos_exact):
        return "3pl-indef-pronoun"
    elif re.search("VNW\(onbep", pos_exact): # Overlaps with previous regex, depends on evaluation order
        return "3sg-indef-pronoun"
    elif re.search("VNW\(pers,pron,.*2.*,getal", pos_exact):
        return "2sg/pl"
    else:
        raise ValueError(f"POS tag not recognized: {pos_exact}")

# Frequencies of forms in subject position
# VNW1     51932 personal pronoun nominative
# VNW19    16244 demonstrative pronoun
# VNW3     14475 personal pronoun standard (nominative or oblique)
# N5        2521 noun
# VNW22      922 indefinite pronoun, singular
# N3         828 noun
# VNW5       792 NOT INCLUDE dialect personal pronoun (tag not finegrained enough per person)
# N1         742 noun
# VNW13      193 NOT INCLUDE betrekkelijk voornaamwoord (subordinate sentences, not what we want)
# SPEC       179 NOT INCLUDE special
# VNW20      171 NOT include aanwijzend voornaamwoord adv-pron. er, daar
# VNW21      157 NOT INCLUDE, is put in front of noun, will count double. aanwijzend voornaamwoord det
# LID        125 NOT INCLUDE, does not distinguish independent use (het regent) from adjective use (het boek). determiner
# TW1         88 NOT INCLUDE, does not distinguish number. counting word
# VNW6        78 NOT INCLUDE person reflexive (mezelf, onszelf)
# VG2         64 NOT INCLUDE voegwoord
# VNW2        61 NOT INCLUDE pronoun oblique case
# VNW26       48 NOT INCLUDE indefinite pronoun al/allebei, does not distinguish number
# WW6         40 substantive use of verb. (het) spelen
# ADJ4        40 substantive use of adjective
# ADJ9        38 NOT include, free use of adjective
# N7          38 proper noun plural
# VNW25       29 NOT include, not always distinguishing number, indef
# BW          24 NOT INClude, bijwoord
# VNW24       18 NOT include, not always distinguishing number, indef
# WW4         15 NOT include, normal infinitive
# TSW         10 NOT include, interjection, ja
#------
# VNW11        8
# VNW14        7
# WW1          6
# ADJ1         5
# ADJ5         3
# ADJ10        3
# VNW27        2 indef dialect
# WW2          2
# VZ2          2
# WW7          2
# VG1          1
# ADJ6         1
# VNW16        1
# ADJ2         1
# VZ1          1
# WW9          1

def get_context(row):
    return row.name

def context_diagnostic(df):
    ### Diagnostic of context for determiner
    df_det =df[(df["tag"]=="LID") & (df["edge"]=="SU")]
    context = df_det.apply(lambda r: f'{r["word"]} {df.loc[int(r.name)+1]["word"]}', axis=1)
    print(context)
    context.to_csv("det.csv")

def compute_frequencies(df, negraheader_df):

    df_filtered = df[(df["tag"].isin(["VNW1", "VNW19", "VNW3", "N5", "VNW22", "N3", "N1", "WW6","ADJ4", "N7"])) & (df["edge"]=="SU")]
    # Add exact POS tags for columns in 'morph' col, using negraheader.txt
    df_filtered_pos_exact = df_filtered.merge(negraheader_df, how="left", on="morph")
    # Merge exact POS tags into person-numbers
    df_filtered_pos_exact["person_number"] = df_filtered_pos_exact["pos_exact"].apply(pos_exact_to_person)
    # print(df[df["edge"]=="SU"]["tag"].value_counts())
    print("Person frequencies:")
    value_counts = df_filtered_pos_exact["person_number"].value_counts()
    print(value_counts)
    n_filtered = len(df_filtered_pos_exact)
    n_filtered_no3sgpl = n_filtered - len(df_filtered_pos_exact[df_filtered_pos_exact["person_number"]=="3sg/pl-demonstr-pronoun"])
    n_3sg = value_counts["3sg"] + value_counts["3sg-demonstr-pronoun"] + value_counts["3sg-indef-pronoun"] + value_counts["3sg-noun"] + value_counts["3sg-adj"] + value_counts["3sg-verb"] 
    print(f"Total tokens: {n_filtered}. Total 3sg: {n_3sg}. Proportion 3sg: {n_3sg/n_filtered}. Proportion 3sg after removing 3sg/pl: {n_3sg/n_filtered_no3sgpl}")
    print("Used forms")
    for name, group in df_filtered_pos_exact.groupby("person_number"):
        print(name)
        print(group["word"].unique())

def main():
    negraheader_df = pd.read_csv(os.path.join(CGN_DIR, "negraheader.txt"), sep="\s+", engine="python", header=None, names=["id", "morph", "pos_exact"], index_col=False, skiprows=lambda x: x<= 78 or x >= 400)
    negraheader_df = negraheader_df.drop(columns=["id"])

    dfs_comp = defaultdict(list)
    for comp in COMPONENTS_ALL:
        # print(f" - {comp}")
        for area in ["vl", "nl"]:
            syn_path = os.path.join(CGN_DIR, f"comp-{comp}", area)
            if not os.path.exists(syn_path):
                # For some components, there is no data
                # for one area, so no vl or nl directory
                continue
            with os.scandir(syn_path) as syn_files:
                for syn_file in syn_files:
                    #print(syn_file.name)
                    df = pd.read_csv(syn_file.path, sep="\s+", engine="python", encoding = "ISO-8859-1", header=None, names=["word","tag","morph","edge","parent","secedge","comment"], index_col=False, on_bad_lines="skip", skiprows=[0,1,2,3,4], comment="%", compression="infer")
                    df = df[~df.word.str.startswith("#")]
                    df = df[~(df.tag.isin(["LET"]))]
                    dfs_comp[comp].append(df)
    # Compute person frequencies per component
    # for comp in COMPONENTS_ALL:
    #     print(f" - {comp}")
    #     df_comp = pd.concat(dfs_comp[comp], ignore_index=True)
    #     compute_frequencies(df_comp, negraheader_df)


    # Take list of dfs for every component together
    dfs_total = itertools.chain.from_iterable(dfs_comp.values())
    df_total = pd.concat(dfs_total, ignore_index=True)
    print(" - Total")
    compute_frequencies(df_total, negraheader_df)

    
    # Filter on (automatic) generalized POS tags VNW1 and VNW 3: personal pronouns either in nominative case or standard case (not clear whether nom or acc)
    # See negraheader.txt
    # Filter on (manual) syntactic tag SU: subject. For now we leave out SUP (provisional subject, specific uses of 'er' and 'het')

if __name__ == "__main__":
    main()
    