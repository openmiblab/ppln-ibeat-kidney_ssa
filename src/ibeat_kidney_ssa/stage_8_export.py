import os
import logging
import argparse
from datetime import date

import pydmr
import pandas as pd

from miblab import pipe

PIPELINE = 'kidney_ssa'

def run(build):

    logging.info("Stage 8 --- Exporting shape features ---")
    dir_measure = os.path.join(build, PIPELINE, 'stage_7_extract_features')
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)
    
    today = date.today().strftime("%Y-%m-%d")

    # Outputs
    dmr_output_file = os.path.join(dir_output, f'NormalizedKidneyShape_{today}')
    long_format_file = os.path.join(dir_output, f'NormalizedKidneyShape_{today}.csv')
    wide_format_file = os.path.join(dir_output, f'NormalizedKidneyShape_{today}_wide.csv')

    # Inputs
    dmr_file = os.path.join(dir_measure, f'all_kidneys.dmr.zip')
    if not os.path.exists(dmr_file):
        return

    # Append parsed biomarkers in the dictionary
    dmr = pydmr.read(dmr_file)
    dmr['columns'] = ['body_part', 'image', 'biomarker_category', 'biomarker']
    for p in dmr['data']:
        parts = p.split('-')
        # For intrinsic markers add image 'mask'
        if len(parts) == 3:
            parts = [parts[0]] + ['mask'] + parts[1:]
        dmr['data'][p] += parts

    # Change PatientIDs to central format
    pars_harmonized = {}
    for p,v in dmr['pars'].items():
        harmonized_id = f"iBE-{p[0].replace('_','')}"
        visit = visit_nr(p[1])
        harmonized_id, visit = fix_exeter_volunteer(harmonized_id, visit)
        pars_harmonized[(harmonized_id, visit, p[2])] = v
    dmr['pars'] = pars_harmonized

    # Save results
    pydmr.write(dmr_output_file, dmr)
    pydmr.pars_to_long(dmr_output_file, long_format_file)
    pydmr.pars_to_wide(dmr_output_file, wide_format_file)

    # Replace column names in long and wide formats
    new_cols = {
        "subject": "harmonized_id",
        "study": "visit_nr",
        "value": "result",
    }
    df = pd.read_csv(long_format_file)
    df.rename(columns=new_cols, inplace=True)
    df.to_csv(long_format_file, index=False)
    
    df = pd.read_csv(wide_format_file)
    df.rename(columns=new_cols, inplace=True)
    df.to_csv(wide_format_file, index=False)

    logging.info(f"Stage 8. Successfully exported shape features")



def visit_nr(value):
    if value == 'Baseline':
        return 0
    if value == 'Followup':
        return 2
    if value[:5] == 'Visit':
        return int(value[5]) - 1
    
    
def fix_exeter_volunteer(harmonized_id, visit_nr):
    # Correct a mistake in ID 
    # Exeter volunteer 3 is the same person as volunteer 1
    # This needs to be removed when the issue is fixed at the source
    if harmonized_id == 'iBE-3128C03':
        harmonized_id = 'iBE-3128C01'
        visit_nr += 2
    return harmonized_id, visit_nr


if __name__=='__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_script(run, BUILD, PIPELINE)
        