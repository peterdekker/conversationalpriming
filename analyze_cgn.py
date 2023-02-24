
# analyze_cgn.py: Standalone script to count persons in Corpus Gesproken Nederlands (CGN, corpus of spoken Dutch)
# Input data expected to be in folder CGN_DIR. By default this is 'data/CGNAnn', with within that the 'Data' folder that contains the whole directory structure.
# See README for more information on installation.

import pandas as pd
import os
from collections import defaultdict
import itertools
import re

CGN_DIR = os.path.join("data", "CGNAnn", "Data", "data", "annot", "text", "syn")

COMPONENT_FACE_TO_FACE = ["a"]
COMPONENTS_INTERACTION = ["a", "b", "c", "d", "e", "f", "g"]
COMPONENTS_ALL = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]

def pos_exact_to_person(pos_exact):
    if re.search("VNW\(pers,pron,.*1.*,ev", pos_exact):
        return "1sg"
    elif re.search("VNW\(pers,pron,.*2.*,ev", pos_exact):
        return "2sg"
    elif re.search("VNW\(pers,pron,.*3.*,ev", pos_exact):
        return "3sg"
    elif re.search("N\(.*ev", pos_exact):
        return "3sg-noun"
    elif re.search("VNW\(onbep", pos_exact):
        return "3sg-indef-pronoun"
    elif re.search("VNW\(aanw.*ev", pos_exact):
        return "3sg-demonstr-pronoun"
    elif re.search("VNW\(aanw", pos_exact): # Overlaps with previous regex, depends on evaluation order
        return "3sg/pl-demonstr-pronoun"
    elif re.search("VNW\(pers,pron,.*1.*,mv", pos_exact):
        return "1pl"
    elif re.search("VNW\(pers,pron,.*2.*,mv", pos_exact):
        return "2pl"
    elif re.search("VNW\(pers,pron,.*3.*,mv", pos_exact):
        return "3pl"
    elif re.search("N\(.*mv", pos_exact):
        return "3pl-noun"
    elif re.search("VNW\(pers,pron,.*2.*,getal", pos_exact):
        return "2sg/pl"
    else:
        raise ValueError(f"POS tag not recognized: {pos_exact}")

# Frequencies of forms in subject position
# VNW1     51932 personal pronoun nominative
# VNW19    16244 aanwijzend
# VNW3     14475 personal pronoun standard
# N5        2521 noun
# VNW22      922 onbepaald voornaamwoord
# N3         828 noun
# VNW5       792 NOT INCLUDE dialect personal pronoun (tag not finegrained enough per person)
# N1         742 noun
# VNW13      193 NOT INCLUDE betrekkelijk voornaamwoord (subordinate sentences, not what we want)
# SPEC       179
# VNW20      171 aanwijzend voornaamwoord adv-pron
# VNW21      157 aanwijzend voornaamwoord det
# LID        125 determiner
# TW1         88 hoofdtelwoord
# VNW6        78
# VG2         64
# VNW2        61
# VNW26       48
# WW6         40
# ADJ4        40
# ADJ9        38
# N7          38
# VNW25       29
# BW          24
# VNW24       18
# WW4         15
# TSW         10
# VNW11        8
# VNW14        7
# WW1          6
# ADJ1         5
# ADJ5         3
# ADJ10        3
# VNW27        2
# WW2          2
# VZ2          2
# WW7          2
# VG1          1
# ADJ6         1
# VNW16        1
# ADJ2         1
# VZ1          1
# WW9          1

def compute_frequencies(df, negraheader_df):
    # print(f"Len: {len(df)}")
    #print(df[df["edge"]=="SU"]["tag"].value_counts())
    df_filtered = df[(df["tag"].isin(["VNW1", "VNW3", "N5", "N3", "N1", "VNW22", "VNW19","VNW20","VNW21"])) & (df["edge"]=="SU")]
    # Add exact POS tags for columns in 'morph' col, using negraheader.txt
    df_filtered_pos_exact = df_filtered.merge(negraheader_df, how="left", on="morph")
    # Merge exact POS tags into person-numbers
    df_filtered_pos_exact["person_number"] = df_filtered_pos_exact["pos_exact"].apply(pos_exact_to_person)
    # print(df[df["edge"]=="SU"]["tag"].value_counts())
    print("Person frequencies:")
    print(df_filtered_pos_exact["person_number"].value_counts())
    # print("Used forms")
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
    