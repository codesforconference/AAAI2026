import pandas as pd

df = pd.read_csv("yt_wb_icd10.csv")
df['icd10'] = df['icd10'].astype(str).str.upper().str.replace('.', '', regex=False)

icd_descriptions = {
    "D75": ("Other and unspecified diseases of blood and blood-forming organs", "Blood Disorders"),
    "D68": ("Other coagulation defects", "Blood Disorders"),
    "D508": ("Deficiency and other anemia", "Blood Disorders"),
    "D729": ("Disorder of white blood cells, unspecified", "Blood Disorders"),
    "D571": ("Sickle-cell disease without crisis", "Blood Disorders"),

    "I00-I99": ("Diseases of the circulatory system", "Circulatory Diseases"),
    "I67": ("Acute cerebrovascular disease", "Circulatory Diseases"),
    "I21": ("Acute myocardial infarction", "Circulatory Diseases"),
    "I74": ("Arterial embolism and thrombosis", "Circulatory Diseases"),
    "I49": ("Other cardiac arrhythmias", "Circulatory Diseases"),
    "I459": ("Conduction disorder, unspecified", "Circulatory Diseases"),
    "I10": ("Hypertension", "Circulatory Diseases"),
    "I568": ("Occlusion and stenosis of other precerebral arteries", "Circulatory Diseases"),
    "I809": ("Phlebitis and thrombophlebitis of unspecified site", "Circulatory Diseases"),
    "I270": ("Primary pulmonary hypertension", "Circulatory Diseases"),

    "K92": ("Other diseases of digestive system", "Digestive Diseases"),
    "K839": ("Disease of biliary tract, unspecified", "Digestive Diseases"),
    "K57": ("Diverticular disease of intestine", "Digestive Diseases"),
    "K29": ("Gastritis and duodenitis", "Digestive Diseases"),
    "K922": ("Gastrointestinal hemorrhage, unspecified", "Digestive Diseases"),
    "K561": ("Peritoneal abscess", "Digestive Diseases"),
    "K51": ("Ulcerative colitis", "Digestive Diseases"),

    "N00-N99": ("Diseases of the genitourinary system", "Genitourinary Diseases"),
    "N179": ("Acute kidney failure, unspecified", "Genitourinary Diseases"),
    "N809": ("Endometriosis, unspecified", "Genitourinary Diseases"),
    "N498": ("Inflammatory disorders of other specified male genital organs", "Genitourinary Diseases"),
    "N70-N77": ("Inflammatory diseases of female pelvic organs", "Genitourinary Diseases"),
    "N959": ("Unspecified menopausal and perimenopausal disorder", "Genitourinary Diseases"),
    "N819": ("Female genital prolapse, unspecified", "Genitourinary Diseases"),

    "M00-M99": ("Diseases of the musculoskeletal system and connective tissue", "Musculoskeletal Diseases"),
    "M1990": ("Unspecified osteoarthritis, unspecified site", "Musculoskeletal Diseases"),
    "M81": ("Osteoporosis without current pathological fracture", "Musculoskeletal Diseases"),
    "M479": ("Spondylosis, unspecified", "Musculoskeletal Diseases"),

    "G00-G99": ("Personal history of other diseases of the nervous system", "Nervous/Sensory History"),
    "H00-H95": ("... and sense organs", "Nervous/Sensory History"),
    "H40": ("Glaucoma", "Nervous/Sensory History"),
    "G20": ("Parkinson's disease", "Nervous/Sensory History"),
    "H33": ("Retinal detachments and breaks", "Nervous/Sensory History"),

    "J00-J99": ("Diseases of the respiratory system", "Respiratory Diseases"),
    "J98": ("Other respiratory disorders", "Respiratory Diseases"),
    "J9690": ("Respiratory failure, unspecified", "Respiratory Diseases"),

    "E00-E89": ("Endocrine, nutritional and metabolic diseases", "Endocrine/Metabolic"),
    "E789": ("Disorder of lipoprotein metabolism, unspecified", "Endocrine/Metabolic"),
    "E118": ("Type 2 diabetes mellitus with unspecified complications", "Endocrine/Metabolic"),
    "E119": ("Type 2 diabetes mellitus without complications", "Endocrine/Metabolic"),
    "M119": ("Crystal arthropathy, unspecified", "Endocrine/Metabolic"),
    "D899": ("Disorder involving the immune mechanism, unspecified", "Endocrine/Metabolic"),
    "E079": ("Disorder of thyroid, unspecified", "Endocrine/Metabolic"),

    "F01-F99": ("Mental, Behavioral and Neurodevelopmental disorders", "Mental Disorders"),
    "F43": ("Adjustment disorders", "Mental Disorders"),
    "F10": ("Alcohol-related disorders", "Mental Disorders"),
    "F40-F41": ("Anxiety disorders", "Mental Disorders"),
    "F90-F91": ("Attention-deficit and disruptive behavior disorders", "Mental Disorders"),
    "F01-F09": ("Delirium, dementia and amnestic disorders", "Mental Disorders"),
    "F80-F89": ("Developmental disorders", "Mental Disorders"),
    "F30-F39": ("Mood disorders", "Mental Disorders"),
    "F60": ("Personality disorders", "Mental Disorders"),
    "F20": ("Schizophrenia", "Mental Disorders"),

    "C81": ("Hodgkin's disease", "Cancers"),
    "C91-C95": ("Leukemias", "Cancers"),
    "C90": ("Multiple myeloma", "Cancers"),
    "C82-C85": ("Non-Hodgkin's lymphoma", "Cancers")
}

def icd_in_range(icd, code_range):
    if '-' in code_range:
        start, end = code_range.split('-')
        return start <= icd <= end
    return icd == code_range

results = []

for icd_code, (desc, cat) in icd_descriptions.items():
    matched = df[df['icd10'].apply(lambda x: icd_in_range(x, icd_code))]
    count = matched['eid'].nunique()
    results.append((icd_code, desc, cat, count))

df_results = pd.DataFrame(results, columns=['icd_code', 'description', 'category', 'num_unique_patients'])
df_results = df_results.sort_values(by='num_unique_patients', ascending=False)

df_results.to_csv("new_icd_counts.csv", index=False)

import pandas as pd
import os

icd_df = pd.read_csv("new_icd_counts.csv")
icd_codes = icd_df['icd_code'].astype(str).str.upper().str.replace('.', '', regex=False).tolist()

df = pd.read_csv("yt_wb_icd10.csv")
df['icd10'] = df['icd10'].astype(str).str.upper().str.replace('.', '', regex=False)

output_dir = "common"
os.makedirs(output_dir, exist_ok=True)

def icd_in_range(icd_code, icd_range):
    if '-' in icd_range:
        start, end = icd_range.split('-')
        return start <= icd_code <= end
    else:
        return icd_code == icd_range

for code in icd_codes:
    matched = df[df['icd10'].apply(lambda x: icd_in_range(x, code))]
    if not matched.empty:
        output_path = os.path.join(output_dir, f"{code}.csv")
        matched.to_csv(output_path, index=False)
        print(f"Saved {len(matched)} rows to {output_path}")
    else:
        print(f"No records found for ICD: {code}")

import os
import pandas as pd

df_full = pd.read_csv("yt_wb_icd10.csv")
df_full['icd10'] = df_full['icd10'].str.upper()

input_dir = "common"
output_dir = "common_with_family"
os.makedirs(output_dir, exist_ok=True)

for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_dir, filename)
        df_disease = pd.read_csv(file_path)

        if 'familyid' not in df_disease.columns:
            continue
        family_ids = df_disease['familyid'].unique()
        df_family = df_full[df_full['familyid'].isin(family_ids)]
        output_path = os.path.join(output_dir, f"with_family_{filename}")
        df_family.to_csv(output_path, index=False)

import pandas as pd
import os
from glob import glob
from tqdm import tqdm

manual_icd_ranges = {
    "C82-C85": ["C82", "C83", "C84", "C85"],
    "C91-C95": ["C91", "C92", "C93", "C94", "C95"],
    "F01-F09": ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09"],
    "F30-F39": ["F30", "F31", "F32", "F33", "F34", "F38", "F39"],
    "F40-F41": ["F40", "F41"],
    "F80-F89": ["F80", "F81", "F82", "F83", "F84", "F88", "F89"],
    "N70-N77": ["N70", "N71", "N72", "N73", "N74", "N75", "N76", "N77"]
}

input_dir = "common_with_family"
output_dir = "common_long"
os.makedirs(output_dir, exist_ok=True)

target_files = [f for f in glob(os.path.join(input_dir, "with_family_*.csv")) if '-' in f]

for file in tqdm(target_files, desc="ICD"):
    icd_code = os.path.basename(file).replace("with_family_", "").replace(".csv", "").upper()
    if icd_code not in manual_icd_ranges:
        print(f"{icd_code}")
        continue

    icd3_list = manual_icd_ranges[icd_code]

    df = pd.read_csv(file)
    df['event_dt'] = pd.to_datetime(df['event_dt'], dayfirst=True, errors='coerce')
    df['icd10'] = df['icd10'].astype(str).str.upper()
    df['icd3'] = df['icd10'].str[:3]
    df['is_target'] = df['icd3'].isin(icd3_list)

    first_dt = df[df['is_target']].groupby('eid')['event_dt'].min().reset_index()
    first_dt.columns = ['eid', 'first_target_dt']
    df = df.merge(first_dt, on='eid', how='left')

    df_filtered = df[df['first_target_dt'].isna() | (df['event_dt'] <= df['first_target_dt'])].copy()
    df_filtered['target'] = (
        (df_filtered['event_dt'] == df_filtered['first_target_dt']) &
        df_filtered['is_target']
    ).astype(int)

    target1 = df_filtered[df_filtered['target'] == 1].drop_duplicates(subset=['eid', 'event_dt', 'icd10'])
    not_target1 = df_filtered[df_filtered['target'] == 0]
    final_df = pd.concat([target1, not_target1], ignore_index=True).sort_values(['eid', 'event_dt'])

    out_path = os.path.join(output_dir, f"common_{icd_code}_for_Lmodel.csv")
    final_df[['eid', 'event_dt', 'icd10', 'target','familyid']].to_csv(out_path, index=False)


