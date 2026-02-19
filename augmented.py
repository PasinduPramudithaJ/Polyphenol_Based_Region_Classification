import pandas as pd
import numpy as np

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
INPUT_FILE = "data/Polyphenol_Primary_Dataset.csv"
OUTPUT_FILE = "data/Polyphenol_Expanded_Dataset.csv"
ROWS_PER_REGION = 40000

np.random.seed(42)

# --------------------------------------------------
# LOAD ORIGINAL DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

regions = df["Region"].unique()

all_generated = []

# --------------------------------------------------
# GENERATE DATA PER REGION
# --------------------------------------------------
for region in regions:
    
    region_df = df[df["Region"] == region]
    
    # Preserve real grade distribution
    grade_distribution = region_df["Grade"].value_counts(normalize=True)
    
    region_rows_count = 0
    
    for grade, proportion in grade_distribution.items():
        
        # rows needed for this grade (proportional)
        rows_needed = int(round(ROWS_PER_REGION * proportion))
        region_rows_count += rows_needed
        
        grade_df = region_df[region_df["Grade"] == grade]
        
        absorb_mean = grade_df["Absorbance"].mean()
        absorb_std = grade_df["Absorbance"].std()
        
        conc_mean = grade_df["Concentration"].mean()
        conc_std = grade_df["Concentration"].std()
        
        dry_mean = grade_df["Dry matter content"].mean()
        dry_std = grade_df["Dry matter content"].std()
        
        caf_mean = grade_df["Caffiene Content"].mean()
        caf_std = grade_df["Caffiene Content"].std()
        
        for i in range(rows_needed):
            
            sample_name = f"{region[:3].upper()}_{grade}_{i+1:05d}"
            
            new_row = {
                "Region": region,
                "Grade": grade,
                "Sample Name": sample_name,
                "Absorbance": round(max(0, np.random.normal(absorb_mean, absorb_std)), 3),
                "Concentration": round(max(0, np.random.normal(conc_mean, conc_std)), 2),
                "Dry matter content": round(max(0, np.random.normal(dry_mean, dry_std)), 2),
                "Caffiene Content": round(max(0, np.random.normal(caf_mean, caf_std)), 2)
                                    if not np.isnan(caf_mean) else np.nan
            }
            
            all_generated.append(new_row)
    
    # 🔥 Fix rounding difference (if total not exactly 40000)
    difference = ROWS_PER_REGION - region_rows_count
    if difference > 0:
        sample_extra = region_df.sample(1).iloc[0]
        for i in range(difference):
            extra_row = sample_extra.to_dict()
            extra_row["Sample Name"] = f"{region[:3].upper()}_EXTRA_{i+1:05d}"
            all_generated.append(extra_row)

# --------------------------------------------------
# FINAL DATASET
# --------------------------------------------------
final_df = pd.DataFrame(all_generated)

print("Total rows generated:", len(final_df))

final_df.to_csv(OUTPUT_FILE, index=False)

print("Saved as:", OUTPUT_FILE)
