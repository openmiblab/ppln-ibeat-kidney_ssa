"""Functions to read and write data in src/data"""

import os
import csv
import re
import json
from pathlib import Path


STUDY_CODE = {
    'Baseline': 'BL',
    'Followup': 'FU',
    'Visit1': 'V1',
    'Visit2': 'V2',
    'Visit3': 'V3',
    'Visit4': 'V4',
    'Visit5': 'V5',
    'Visit6': 'V6',
    'Visit7': 'V7',
}

SERIES_CODE = {
    "normalized_kidney_right": 'R',
    "normalized_kidney_left": 'L',
}


def sort_kidney_masks(kidney_masks):
    # Get labels
    kidney_labels = [label_db_series(k) for k in kidney_masks]
    # Sort labels and masks by labels
    sorted_pairs = sorted(zip(kidney_labels, kidney_masks))
    # Extract sorted masks and labels
    kidney_labels_sorted = [label for label, _ in sorted_pairs]
    kidney_masks_sorted  = [mask  for _, mask in sorted_pairs]
    return kidney_masks_sorted, kidney_labels_sorted


def sort_kidney_npz(kidney_masks):
    """Sort masks and labels by the labels"""
    # Get labels
    kidney_labels = [label_npz_dbfile(f) for f in kidney_masks]
    # Sort labels and masks by labels
    sorted_pairs = sorted(zip(kidney_labels, kidney_masks))
    # Extract sorted masks and labels
    kidney_labels_sorted = [label for label, _ in sorted_pairs]
    kidney_masks_sorted  = [mask  for _, mask in sorted_pairs]
    return kidney_masks_sorted, kidney_labels_sorted

def sort_kidney_series(kidney_masks):
    kidney_labels = []
    for mask_series in kidney_masks:
        patient = mask_series[1]
        study = mask_series[2][0]
        kidney = 'L' if 'left' in mask_series[3][0] else 'R'
        label = f"{patient}-{STUDY_CODE[study]}-{kidney}"
        kidney_labels.append(label)
    kidney_masks_sorted = [v for _, v in sorted(zip(kidney_labels, kidney_masks))]
    return kidney_masks_sorted
    
def relpath_npz_dbfile(file):
    
    # Get subdirectories
    studypath = os.path.dirname(file)
    patientpath = os.path.dirname(studypath)

    # Get names of patient, study and series
    series = os.path.basename(file)
    study = os.path.basename(studypath)
    patient = os.path.basename(patientpath)

    return os.path.join(patient, study, series)

def parse_npz_dbfile(file) -> dict:

    # Parse file path
    patient = os.path.basename(os.path.dirname(os.path.dirname(file)))
    patient = patient.split('__')[-1]
    study = os.path.basename(os.path.dirname(file))
    study = study.split('__')[-1]
    kidney = os.path.basename(file)[:-4]
    kidney = kidney.split('__')[-1]

    # Construct label 
    if STUDY_CODE[study] not in ['V1', 'BL']:
        raise ValueError(f'Series {patient}, {study}, {kidney} is not a baseline series')
    k = 'L' if 'left' in kidney else 'R'
    label = f"{patient}-{k}"

    # Construct dict
    value = {
        'PatientID': patient, 
        'StudyDescription': study, 
        'SeriesDescription': kidney,
        # TODO return StudyID and SeriesNumber
    }
    return label, value


def label_db_series(series):
    patient = series[1]
    study = series[2][0]
    kidney = 'L' if 'left' in series[3][0] else 'R'  
    if STUDY_CODE[study] not in ['V1', 'BL']:
        raise ValueError(f'Series {series[1:]} is not a baseline series')
    # label = f"{patient}-{STUDY_CODE[study]}-{kidney}"
    label = f"{patient}-{kidney}" 
    return label

def label_npz_dbfile(file) -> dict:

    # Parse file path
    patient = os.path.basename(os.path.dirname(os.path.dirname(file)))
    patient = patient.split('__')[-1]
    study = os.path.basename(os.path.dirname(file))
    study = study.split('__')[-1]
    kidney = os.path.basename(file)[:-4]
    kidney = kidney.split('__')[-1]

    # Construct label
    if STUDY_CODE[study] not in ['V1', 'BL']:
        raise ValueError(f'Series {patient}, {study}, {kidney} is not a baseline series')
    # label = f"{patient}-{STUDY_CODE[study]}-{k}"
    kidney = 'L' if 'left' in kidney else 'R'
    label = f"{patient}-{kidney}" 
    return label


def dixon_record():
    record = os.path.join(os.getcwd(), 'src', 'data', 'dixon_data.csv')
    with open(record, 'r') as file:
        reader = csv.reader(file)
        record = [row for row in reader]
    return record


def dixon_series_desc(record, patient, study):
    for row in record:
        if row[1] == patient:
            if row[2]==study:
                return row[5]
    raise ValueError(
        f'Patient {patient}, study {study}: not found in src/data/dixon_data.csv'
    )


def parse_cluster_file(filepath, save_json=True, json_path=None):
    """
    Parse a cluster text file into a dictionary and optionally save as JSON.

    Parameters
    ----------
    filepath : str
        Path to the text file containing cluster definitions.
    save_json : bool, optional
        Whether to save the resulting dictionary as a JSON file. Default is True.
    json_path : str, optional
        Path to save the JSON output. If None, saves next to the input file
        with the same base name but '.json' extension.

    Returns
    -------
    dict
        Dictionary mapping cluster numbers (int) -> list of subject IDs (str).

    Example
    -------
    >>> clusters = parse_cluster_file("clusters.txt")
    >>> print(clusters[1][:5])
    ['1128_008', '1128_017', '1128_025', '1128_029', '1128_049']
    """
    clusters = {}
    current_cluster = None
    buffer = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Detect start of a new cluster
            match = re.match(r'^Cluster\s+(\d+)\s*\(.*?\):', line, re.IGNORECASE)
            if match:
                # Save previous cluster if needed
                if current_cluster and buffer:
                    ids = [x.strip() for x in re.split(r'[,\s]+', ' '.join(buffer)) if x.strip()]
                    clusters[current_cluster] = ids
                    buffer = []

                current_cluster = int(match.group(1))
                continue

            # Accumulate subject lines
            if line:
                buffer.append(line)

        # Save last cluster
        if current_cluster and buffer:
            ids = [x.strip() for x in re.split(r'[,\s]+', ' '.join(buffer)) if x.strip()]
            clusters[current_cluster] = ids

    # --------------------------------------------------------------------------
    # Save to JSON if requested
    # --------------------------------------------------------------------------
    if save_json:
        if json_path is None:
            base, _ = os.path.splitext(filepath)
            json_path = f"{base}.json"

        # Convert keys to strings for JSON compatibility
        with open(json_path, 'w', encoding='utf-8') as jf:
            json.dump(clusters, jf, indent=4)

        print(f"✅ Saved cluster dictionary to: {json_path}")

    return clusters


if __name__=='__main__':

    cluster_ids = "C:\\Users\\md1spsx\\Documents\\GitHub\\iBEAt-pipeline-kidneyvol\\src\\data\\Cluster_IDs.txt"
    parse_cluster_file(cluster_ids, save_json=True)