import argparse
import pandas as pd
import numpy as np
from mesa.batchrunner import BatchRunner
import os
import util
import seaborn as sns
import matplotlib.pyplot as plt

from model import Model
from config import model_params_script, evaluation_params, bool_params, string_params, OUTPUT_DIR, IMG_FORMAT, LAST_N_STEPS_END_GRAPH, PERSONS

communicated_stats = ["prop_innovative_1sg_innovating_avg", "prop_innovative_1sg_conservating_avg", "prop_innovative_1sg_total_avg", "prop_innovative_2sg_innovating_avg", "prop_innovative_2sg_conservating_avg", "prop_innovative_2sg_total_avg", "prop_innovative_3sg_innovating_avg", "prop_innovative_3sg_conservating_avg", "prop_innovative_3sg_total_avg"]
internal_stats = ["prop_innovative_1sg_innovating_internal", "prop_innovative_1sg_conservating_internal", "prop_innovative_1sg_total_internal", "prop_innovative_2sg_innovating_internal", "prop_innovative_2sg_conservating_internal", "prop_innovative_2sg_total_internal", "prop_innovative_3sg_innovating_internal", "prop_innovative_3sg_conservating_internal", "prop_innovative_3sg_total_internal"]
dominant_stats = ["prop_1sg_conservating_dominant", "prop_2sg_conservating_dominant", "prop_3sg_conservating_dominant", "prop_1sg_innovating_dominant", "prop_2sg_innovating_dominant", "prop_3sg_innovating_dominant", "prop_1sg_total_dominant", "prop_2sg_total_dominant", "prop_3sg_total_dominant"]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def params_print(params):
    return "".join([f"{k}: {v}   " for k, v in params.items()])


def create_graph(run_data, variable_param, stats, mode, output_dir):
    print(f"{mode}:")
    course_df = get_course_df(run_data, variable_param, stats, mode, output_dir)

    plots(variable_param, stats, mode, output_dir, course_df)

def create_contrast_persons_graph(run_data, stats, mode, output_dir):
    print(f"{mode}:")
    course_df = get_course_df(run_data, None, stats, mode, output_dir)

    stats_1sg_3sg_contrast = [stat for stat in stats if ("1sg" in stat or "3sg" in stat) and "total" not in stat]
    plot_contrast_persons_graph(course_df, stats_1sg_3sg_contrast, mode, output_dir, "1sg-3sg-contrast")

def plots(variable_param, stats, mode, output_dir, course_df):
    print("Plot graph 1sg-3sg.")
    # Comparison 1sg-3sg
    stats_1sg_3sg = [stat for stat in stats if "1sg" in stat or "3sg" in stat]
    plot_graph(course_df, variable_param, stats_1sg_3sg, mode, output_dir, "1sg-3sg")

    for person in ["1sg","3sg"]: # For now, don't plot 2sg because it is the same as 1sg
        print(f"Plot graph {person}.")
        stats_person = [stat for stat in stats if person in stat] # if person (eg. 1sg) is substring of stat name
        plot_graph(course_df, variable_param, stats_person, mode, output_dir, person)

def get_course_df(run_data, variable_param, stats, mode, output_dir):
    print("Getting course df.")
    iteration_dfs = []
    for i, row in run_data.iterrows():
        iteration_df = row["datacollector"].get_model_vars_dataframe()[stats]
        if variable_param: # in contrast_persons mode no variable param
            iteration_df[variable_param] = row[variable_param]
        # Drop all rows with index 0, since this is a logging artefact
        iteration_df = iteration_df.drop(0)
        iteration_dfs.append(iteration_df)
    course_df = pd.concat(iteration_dfs)
    # Old index (with duplicates because of different param settings and runs) becomes explicit column 'timesteps'
    course_df = course_df.reset_index().rename(columns={"index":"timesteps"})
    var_label = f"{variable_param}-" if variable_param else ""
    course_df.to_csv(os.path.join(output_dir, f"{var_label}{mode}-raw.csv"))
    return course_df


def plot_graph(course_df, variable_param, stats, mode, output_dir, label):
    df_melted = course_df.melt(id_vars=["timesteps",variable_param], value_vars = stats, value_name = "proportion innovative forms", var_name="statistic")
    sns.lineplot(data=df_melted, x="timesteps", y="proportion innovative forms", hue="statistic", style=variable_param)

    # Label is usually person (e.g. 1sg)
    plt.savefig(os.path.join(output_dir, f"{variable_param}-{label}-{mode}.{IMG_FORMAT}"), format=IMG_FORMAT, dpi=300)
    plt.clf()

def plot_contast_persons_graph(course_df, stats, mode, output_dir, label):
    df_melted = course_df.melt(id_vars=["timesteps"], value_vars = stats, value_name = "proportion innovative forms", var_name="statistic")
    print(df_melted)
    sns.lineplot(data=df_melted, x="timesteps", y="proportion innovative forms", style="statistic")

    # Label is usually person (e.g. 1sg)
    plt.savefig(os.path.join(output_dir, f"{label}-{mode}.{IMG_FORMAT}"), format=IMG_FORMAT, dpi=300)
    plt.clf()

def evaluate_model(fixed_params, variable_params, iterations, steps):
    print(f"- Running batch: {iterations} iterations of {steps} steps")
    if variable_params:
        print(f"  Variable parameters: {params_print(variable_params)}")
    print(f"  Fixed parameters: {params_print(fixed_params)}")

    batch_run = BatchRunner(
        Model,
        variable_params,
        fixed_params,
        iterations=iterations,
        max_steps=steps,
        model_reporters={"datacollector": lambda m: m.datacollector}
    )

    batch_run.run_all()


    run_data = batch_run.get_model_vars_dataframe()


    return run_data


def main():
    parser = argparse.ArgumentParser(description='Run agent model from terminal.')
    model_group = parser.add_argument_group('model', 'Model parameters')
    for param in model_params_script:
        model_group.add_argument(f"--{param}", nargs="+",
                                 type=str2bool if param in bool_params else float)
    evaluation_group = parser.add_argument_group('evaluation', 'Evaluation parameters')
    for param in evaluation_params:
        if param in bool_params:
            evaluation_group.add_argument(f'--{param}', action='store_true')
        elif param in string_params:
            evaluation_group.add_argument(f'--{param}', type=str, default=evaluation_params[param])
        else:
            evaluation_group.add_argument(f"--{param}", nargs="+", type=int, default=evaluation_params[param])

    # Parse arguments
    args = vars(parser.parse_args())
    iterations = args["iterations"]
    steps = args["steps"]
    plot_from_raw = args["plot_from_raw"]
    plot_from_raw_on = args["plot_from_raw"] != ""
    contrast_persons = args["contrast_persons"]

    
    output_dir_custom = OUTPUT_DIR
    if args["runlabel"] != "":
        output_dir_custom = f'{OUTPUT_DIR}-{args["runlabel"]}'
    util.create_output_dir(output_dir_custom)

    if plot_from_raw_on:
        course_df_import = pd.read_csv(plot_from_raw, index_col=0)
        print(course_df_import)
        # Bit of a hack, retrieving stats from column names, assuming they start with prop
        stats_import = [col for col in course_df_import.columns if col.startswith("prop")]
        # Give "from_raw" as mode, because we dont know with which mode the file was generated
        # Assume 'repeats' was var_param
        plots("repeats", stats_import, "from_raw", output_dir_custom, course_df_import)
    else:
        # If not plot_from_raw, perform new run of the model. 2 options:
        # - Contrast persons: plot 1sg vs 3sg, no variable param. Every supplied param is a fixed param
        if contrast_persons:
            print("Contrast persons")
            assert len(iterations) == 1
            assert len(steps) == 1
            iterations_setting = iterations[0]
            steps_setting = steps[0]
            # Take only first item from every parameter, which is list because of nargs=+
            given_model_params = {k: v[0] for k, v in args.items() if k in model_params_script and v is not None}
            fixed_params = {k: (v if k not in given_model_params else given_model_params[k]) for k, v in model_params_script.items()}
            print(fixed_params)
            run_data = evaluate_model(fixed_params, None,
                                        iterations_setting, steps_setting)
            print(run_data)
            create_contrast_persons_graph(run_data, None,
                                stats=communicated_stats, mode="communicated", output_dir=output_dir_custom)
            
            # create_graph(run_data, var_param,
            #                     stats=internal_stats, mode="internal", output_dir=output_dir_custom)
        else:
            # - Regular run:
            # Try variable parameters one by one, while keeping all of the other parameters fixed
            variable_params = {k: v for k, v in args.items() if k in model_params_script and v is not None}
            assert len(variable_params)>0
            print(f"Evaluating iterations {iterations} and steps {steps}")
            for var_param, var_param_settings in variable_params.items():
                assert len(iterations) == 1
                assert len(steps) == 1
                iterations_setting = iterations[0]
                steps_setting = steps[0]
                fixed_params = {k: v for k, v in model_params_script.items() if k != var_param}
                run_data = evaluate_model(fixed_params, {var_param: var_param_settings},
                                            iterations_setting, steps_setting)
                # create_graph(run_data, var_param,
                #                     stats=dominant_stats, mode="dominant", output_dir=output_dir_custom)

                create_graph(run_data, var_param,
                                    stats=communicated_stats, mode="communicated", output_dir=output_dir_custom)
                
                create_graph(run_data, var_param,
                                    stats=internal_stats, mode="internal", output_dir=output_dir_custom)
        


if __name__ == "__main__":
    main()
