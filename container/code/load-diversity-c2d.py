import pandas
import gc
import os
import numpy as np
import math
import magic
import sys
import matplotlib.pyplot as pyplot
import scipy.stats as stats

INPUT_DIR = "/data/inputs"                 # location of input data (regular or zipped CSV) with load user load profiles
OUTPUT_DIR = "/data/outputs"               # location of output data (PDF)
LOG = "/data/logs/load-diversity-c2d.log"  # Log file location, can be used in the log method
SAMPLESIZE = 50 #100                       # Number of independent samples to use to estimate averages
NUMBOOTSTRAP = 1000 #2000                  # Number of bootstrap resampling steps
STEPFACTOR = 6 #3                          # Aggregation level step size (multiplicative, i.e. 3 gives 1,3,9,...)


def log(msg, exit_code=-1):
    print(msg)
    # print(msg, file=open(LOG, 'a'))

    if exit_code >= 0:
        sys.exit(exit_code)

def memory_stats(msg):
    log(msg)
    os.system("cat /proc/meminfo | grep Mem")

memory_stats("Start of script")

## Load data
input_type = None
input_file = None


for basedir, subdir, files in os.walk(INPUT_DIR):
    if input_type is not None:
        break
    for file in files:
        input_file = os.path.join(basedir, file)
        mime_type = magic.from_file(input_file)
        if "csv" in mime_type.lower():
            log(f"Using csv data file {input_file}")
            input_type = "csv"
            break
        elif "zip" in mime_type.lower():
            log(f"Using zip data file {input_file}")
            input_type = "zip"
            break
        else:
            log(f"Skipping {input_file} (not a .CSV or .CSV.ZIP file. Mime type: ${mime_type.lower()})")


dataframe = None
if input_type == "csv":
    log(f"Start reading {input_file} csv file...")
    dataframe = pandas.read_csv(input_file, index_col=0, parse_dates=True, dtype='float32')
    log(f"Finished reading {input_file} csv file")
elif input_type == "zip":
    # Note that this branch of the expression is currently not being used because of an issue on the C2D when reading zips directly
    log(f"Start reading {input_file} csv file from zip...")
    dataframe = pandas.read_csv(input_file, compression=input_type, index_col=0, parse_dates=True, dtype='float32')
    log(f"Finished reading {input_file} csv file from zip")
else:
    log(f"Error: no .CSV or .CSV.ZIP file found in {INPUT_DIR}", 1) # We exit with an exit code in this case

gc.collect()
memory_stats("after reading csv")


## Sampling and diversity factor calculation

# determine the number of aggregation levels according to the size of the dataset and the STEPFACTOR
aggregation_steps = 1 + math.floor(math.log(len(dataframe.columns)) / math.log(STEPFACTOR))
# calculate the aggregation levels, starting at 1
aggregation_levels = [STEPFACTOR ** i for i in range(aggregation_steps)]

log(f"Sampling diversity factors for {len(aggregation_levels)} aggregation levels:")
# initialise result array
diversity_factor_array = np.zeros((len(aggregation_levels), SAMPLESIZE))
# for every aggregation level and sample count
for agg_step, households in enumerate(aggregation_levels):
    gc.collect()
    memory_stats(f"agg step: {agg_step}")
    for i in range(SAMPLESIZE):
        # sample 'households' with replacement
        sample = dataframe.sample(n=households, axis='columns', replace=True)
        # calculate diversity factor
        peak_aggregate_demand = sample.sum(axis='columns').max()
        sum_of_peak_demand = sample.max().sum()
        if peak_aggregate_demand > 1e-5:
            diversity_factor = sum_of_peak_demand / peak_aggregate_demand
        else:
            # define DF=1.0 when peak aggregate demand is zero or negative
            diversity_factor = 1.0
        # store value in array
        diversity_factor_array[agg_step, i] = diversity_factor
log(f"Sampling complete.")

dataframe = None
gc.collect()
memory_stats("after sampling")

## Postprocessing

log(f"Postprocessing results:")
# calculate sample average for each aggregation level
average_diversity_factor = diversity_factor_array.mean(axis=1)

# identify sample sets with non-identical values (usually all other than aggregation level = 1)
bootstrap_filter = (np.equal(diversity_factor_array[:, 0:1], diversity_factor_array).all(axis=1) == False)
# apply BCa bootstrap analysis
bootstrap_stats = stats.bootstrap((diversity_factor_array[bootstrap_filter],), np.mean, n_resamples=NUMBOOTSTRAP, axis=1, vectorized=True)

# extract bootstrap standard errors (and use zero where all samples are identical)
standard_error = np.zeros(aggregation_steps)
standard_error[bootstrap_filter] = bootstrap_stats.standard_error

# extract absolute up / down errors
diversity_factor_errors = np.zeros((2, aggregation_steps))
diversity_factor_errors[0, bootstrap_filter] = average_diversity_factor[bootstrap_filter] - bootstrap_stats.confidence_interval.low
diversity_factor_errors[1, bootstrap_filter] = bootstrap_stats.confidence_interval.high - average_diversity_factor[bootstrap_filter]

log(f"Postprocessing complete.")

## Output

def round_se(x, se):
    """
    Helper function to print value with its standard error, in decimal notation

    Parameters
    ----------
    x : float
        value to be printed
    se : float
        standard error of x

    Returns
    -------
    string
        Output of type "x (se)" with appropriate number of digits
    """
    if (se > 0):
        # determine number of digits to the left of the decimal point for x and se
        digits_x = int(math.floor(math.log10(abs(x))))
        digits_se = int(math.floor(math.log10(abs(se))))
        # compute the number of significant digits for x
        digits = digits_x - digits_se + 2
        # construct return string
        return f"{np.format_float_positional(x, precision=digits, fractional=False, trim='k')} " + \
            f"({np.format_float_positional(se, precision=2, fractional=False, trim='k')})"
    elif (se == 0):
        # Precision is infinite, so report using standard precision
        return np.format_float_positional(x, trim='k') + " (0)"
    else:
        # se cannot be negative
        raise ValueError


# print tabular output
log("    #      diversity factor (standard error)")
for idx, households in enumerate(aggregation_levels):
    log(f"{households:5}      {round_se(average_diversity_factor[idx], standard_error[idx])}")

# plot results

pyplot.errorbar(aggregation_levels, average_diversity_factor, yerr=diversity_factor_errors, fmt='.--')
pyplot.xscale('log')
pyplot.ylim(0, None)
pyplot.xlabel('aggregation level [# of connections]')
pyplot.ylabel('diversity factor')
pyplot.title('diversity factors with 95% confidence intervals')
# save a PDF copy
pyplot.savefig(os.path.join(OUTPUT_DIR, 'diversity_factor.pdf'))

log("Done", 0)
