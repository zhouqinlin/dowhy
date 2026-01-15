import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ==========================================
# Configuration
# ==========================================
RESULTS_DIR = "exp_result"

# Map readable experiment names to the script prefixes used in CSV filenames
EXPERIMENTS = {
    "Stop Promo (0)": "rossmann_promo0",
    "Start Promo (1)": "rossmann_promo1",
    "Do Nothing": "rossmann_do_nothing"
}

# Plot settings
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

def load_data(file_prefix):
    """Loads proposed and conventional aggregated results."""
    prop_path = f"{RESULTS_DIR}/{file_prefix}_proposed_method_aggregated.csv"
    conv_path = f"{RESULTS_DIR}/{file_prefix}_conventional_method_aggregated.csv"
    
    if not os.path.exists(prop_path) or not os.path.exists(conv_path):
        print(f"Warning: Files for {file_prefix} not found. Skipping.")
        return None, None

    # Load data (assuming Store is the index or first column)
    df_prop = pd.read_csv(prop_path)
    df_conv = pd.read_csv(conv_path)
    
    # Ensure consistency
    if 'Store' not in df_prop.columns:
        df_prop['Store'] = df_prop.index
    if 'Store' not in df_conv.columns:
        df_conv['Store'] = df_conv.index
        
    return df_prop, df_conv

def calculate_metrics(df, label):
    """Calculates Lift (Change in Sales) and % Change."""
    # Lift = Counterfactual - Actual
    df['Lift'] = df['POST_Sales'] - df['Sales']
    df['Lift_Percentage'] = (df['Lift'] / df['Sales']) * 100
    df['Method'] = label
    return df

# ==========================================
# Main Analysis
# ==========================================
all_results = []
summary_stats = []

for exp_name, prefix in EXPERIMENTS.items():
    df_prop, df_conv = load_data(prefix)
    
    if df_prop is not None:
        # Calculate metrics for both methods
        df_prop = calculate_metrics(df_prop, "Proposed")
        df_conv = calculate_metrics(df_conv, "Conventional (DoWhy)")
        
        # Combine for analysis
        combined = pd.concat([df_prop[['Store', 'Sales', 'POST_Sales', 'Lift', 'Lift_Percentage', 'Method']], 
                              df_conv[['Store', 'Sales', 'POST_Sales', 'Lift', 'Lift_Percentage', 'Method']]])
        combined['Experiment'] = exp_name
        all_results.append(combined)

        # Calculate Summary Stats for this experiment
        # Mean Absolute Error (MAE) is useful for "Do Nothing", Mean Lift for others
        mae_prop = np.mean(np.abs(df_prop['Lift']))
        mean_lift_prop = df_prop['Lift'].mean()
        
        mae_conv = np.mean(np.abs(df_conv['Lift']))
        mean_lift_conv = df_conv['Lift'].mean()
        
        summary_stats.append({
            "Experiment": exp_name,
            "Prop_Mean_Lift": mean_lift_prop,
            "Conv_Mean_Lift": mean_lift_conv,
            "Prop_MAE": mae_prop,
            "Conv_MAE": mae_conv
        })

final_df = pd.concat(all_results)
summary_df = pd.DataFrame(summary_stats)

# ==========================================
# Visualization
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Analysis of Causal Interventions: Sales Impact', fontsize=16)

# 1. Average Sales Impact (Lift) by Experiment
sns.barplot(data=final_df, x="Experiment", y="Lift", hue="Method", ax=axes[0, 0], errorbar="sd")
axes[0, 0].set_title("Average Impact on Sales (Lift = POST - PRE)")
axes[0, 0].set_ylabel("Change in Sales ($)")
axes[0, 0].axhline(0, color='black', linewidth=1)

# 2. Percentage Change Distribution
sns.boxplot(data=final_df, x="Experiment", y="Lift_Percentage", hue="Method", ax=axes[0, 1])
axes[0, 1].set_title("Distribution of % Sales Change")
axes[0, 1].set_ylabel("% Change")
axes[0, 1].axhline(0, color='black', linewidth=1)

# 3. "Do Nothing" Analysis (Actual vs Predicted)
# We expect points to lie on the diagonal line (y=x)
do_nothing_data = final_df[final_df['Experiment'] == "Do Nothing"]
sns.scatterplot(data=do_nothing_data, x="Sales", y="POST_Sales", hue="Method", style="Method", s=100, ax=axes[1, 0])
# Add diagonal reference line
min_val = min(do_nothing_data['Sales'].min(), do_nothing_data['POST_Sales'].min())
max_val = max(do_nothing_data['Sales'].max(), do_nothing_data['POST_Sales'].max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Reconstruction")
axes[1, 0].set_title("'Do Nothing' Sanity Check: Actual vs Counterfactual")
axes[1, 0].legend()

# 4. Summary Table in Plot
axes[1, 1].axis('off')
table_data = summary_df[['Experiment', 'Prop_Mean_Lift', 'Conv_Mean_Lift']].round(2)
table_data.columns = ['Exp', 'Proposed Lift', 'Conv Lift']
table = axes[1, 1].table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='center')
table.scale(1, 2)
table.set_fontsize(14)
axes[1, 1].set_title("Summary Statistics (Mean Sales Change)")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/analysis_summary.png")
plt.show()

# ==========================================
# Textual Analysis & Explanation
# ==========================================
print("\n" + "="*50)
print("EXPERIMENT ANALYSIS REPORT")
print("="*50)

# 1. Sanity Check Analysis
dn_stats = summary_df[summary_df['Experiment'] == "Do Nothing"].iloc[0]
print(f"\n[1] SANITY CHECK (Do Nothing Experiment)")
print(f"Goal: POST_Sales should equal Sales (Lift ≈ 0).")
print(f" - Proposed Method Mean Error (MAE): {dn_stats['Prop_MAE']:.2f}")
print(f" - Conventional Method Mean Error (MAE): {dn_stats['Conv_MAE']:.2f}")

# 2. Stop Promo Analysis
sp_stats = summary_df[summary_df['Experiment'] == "Stop Promo (0)"].iloc[0]
print(f"\n[2] STOP PROMO EFFECT (Hypothesis: Sales Decrease)")
print(f" - Proposed Method Impact: {sp_stats['Prop_Mean_Lift']:.2f} (Negative is expected)")
print(f" - Conventional Method Impact: {sp_stats['Conv_Mean_Lift']:.2f}")

if sp_stats['Prop_Mean_Lift'] < -dn_stats['Prop_MAE']:
    print(" >> RESULT: Significant drop in sales detected. The intervention worked.")
else:
    print(" >> RESULT: Change is within the margin of error (noise). Effect is weak or model is noisy.")

# 3. Start Promo Analysis
stp_stats = summary_df[summary_df['Experiment'] == "Start Promo (1)"].iloc[0]
print(f"\n[3] START PROMO EFFECT (Hypothesis: Sales Increase)")
print(f" - Proposed Method Impact: {stp_stats['Prop_Mean_Lift']:.2f} (Positive is expected)")
print(f" - Conventional Method Impact: {stp_stats['Conv_Mean_Lift']:.2f}")

print("\n" + "="*50)
print(f"Analysis plots saved to: {RESULTS_DIR}/analysis_summary.png")