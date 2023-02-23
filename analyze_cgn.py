
# analyze_cgn.py: Standalone script to count persons in Corpus Gesproken Nederlands (CGN, corpus of spoken Dutch)
# Input data expected to be in folder CGN_DIR. By default this is 'data/CGNAnn', with within that the 'Data' folder that contains the whole directory structure.
# See README for more information on installation.

import pandas as pd
import os

CGN_DIR = os.path.join("data", "CGNAnn", "Data", "data", "annot", "text", "syn")

COMPONENT_A = ["a"]
COMPONENTS_ALL = ["a", "b", "c", "d", "e", "f", "g"]

def main():
    for comp in COMPONENT_A:
        print(f" - {comp}")
        for area in ["vl", "nl"]:
            syn_path = os.path.join(CGN_DIR, f"comp-{comp}", area)
            if not os.path.exists(syn_path):
                # For some components, there is no data
                # for one area, so no vl or nl directory
                continue
            with os.scandir(syn_path) as syn_files:
                for syn_file in syn_files:
                    print(syn_file.name)
                    df = pd.read_csv(syn_file.path, sep="\t", encoding = "ISO-8859-1", on_bad_lines="skip", skiprows=[0,1,2,4], compression="infer")
                    print(df[(df["tag"].isin(["VNW1", "VNW3"])) & (df["edge"]=="SU")])
                    print(df[df["edge"]=="SU"]["tag"].value_counts())

    
    # Filter on (automatic) generalized POS tags VNW1 and VNW 3: personal pronouns either in nominative case or standard case (not clear whether nom or acc)
    # See negraheader.txt
    # Filter on (manual) syntactic tag SU: subject. For now we leave out SUP (provisional subject, specific uses of 'er' and 'het')

if __name__ == "__main__":
    main()
    