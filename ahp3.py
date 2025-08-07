import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import io
import csv
import seaborn as sns


testo_introduzione = r"""
# AHP-express Decision Tool for DLM comparison with Multi-Interviews Support

This **tool** is designed to support anyone who needs to **select**, **evaluate**, 
or **prioritize** *Data Lifecycle Models* (DLM) without necessarily having 
a specialized background in decision methodologies or AHP. 
Here's a brief overview of how it works:

1. **AHP and AHP-express**  
   - The Analytic Hierarchy Process (**AHP**) is a multi-criteria decision-making 
     method that involves pairwise comparisons of the elements of a problem 
     (criteria, sub-factors, alternatives) to calculate priorities.  
   - **AHP-express** is a **simplified version** of traditional AHP (Saaty) 
     that drastically reduces the number of required comparisons.  
     Instead of comparing *all against all*, it selects a **"base" element** 
     (dominant) and evaluates only *"base vs. others"*. 

2. **Macro-categories with Non-linear Scaling**  
   - Before analyzing sub-factors in detail, there are *macro-categories* 
     (in our case, "A" and "B"), such as the question:
     "Are we managing a project with a large amount of data (A) or not (B)?"
   - Each macro-category can be assigned a **scaling factor** that affects
     the priorities of sub-factors belonging to that category in a non-linear way.
   - The scaling is applied through a diagonal matrix multiplication before normalization.

3. **Sub-factors**  
   - For each macro-category, **sub-factors** (more specific criteria) 
     that characterize the DLMs are defined.  
   - These are also compared using the AHP-express logic, 
     selecting a base sub-factor and comparing it with the others.

4. **Multi-interviews and geometric mean**  
   - If multiple people/experts contribute to the evaluation (e.g., different stakeholders), 
     *separate interviews* are conducted, resulting in multiple sets of comparisons.  
   - To combine all evaluations into a single final result, we use the 
     **geometric mean** of each pairwise comparison (as recommended by Saaty for AHP).

5. **DLM and final scores**  
   - Once the priorities of the sub-factors are calculated with proper macro-category scaling, 
     we can assign each DLM a **total score** 
     by computing the weighted sum of the DLM values for each sub-factor.  
   - The tool displays the **final ranking** and provides summary charts 
     (a *Bar Chart* and a *Radar Chart*) for better result interpretation.

**Getting Started**:  
- Upload an *Excel/CSV* file containing the DLMs and their sub-factors.  
- Set the number of interviews, macro-category assignments and scaling factors.
- For each interview, enter the comparisons "base vs. others".  
- Press the button to calculate the final results.
"""


def carica_dlm_da_file(uploaded_file):
    """
    This function is responsible for loading the DLM data from an excel/csv file separated by different separators
    Carica un file Excel o CSV con possibili separatori diversi.
    """
    filename = uploaded_file.name.lower()
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            return df
        except Exception as e:
            st.error(f"Error reading Excel: {e}")
            return None
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        return df
    except Exception as e:
        st.error(f"Attempt 1 (autodetect) failed: {e}")
        try:
            uploaded_file.seek(0)
            rawdata = uploaded_file.read(2048).decode('utf-8', errors='replace')
            uploaded_file.seek(0)
            dialect = csv.Sniffer().sniff(rawdata, delimiters=[',', ';', '\t', '|'])
            sep = dialect.delimiter
            df = pd.read_csv(uploaded_file, sep=sep)
            return df
        except Exception as e2:
            st.error(f"Attempt 2 (csv.Sniffer) failed: {e2}")
            return None


def normalize_vector(v):
    """Function to normalize a vector (sum to 1)"""
    s = sum(v)
    if s == 0:
        return [0] * len(v)
    return [x / s for x in v]


def calculate_ahp_express_prior(base_index, values):
    """
    CORRECTED: This function calculates the priority vector according to AHP-express
    The correct formula for AHP-express is:
    - For the base element: pr_base = 1 / Σ_k (1/a(base,k))
    - For other elements j: pr_j = (1/a(base,j)) / Σ_k (1/a(base,k))
    
    Where a(base,j) is the comparison value of base vs j
    """
    # Calculate reciprocals for all elements
    reciprocals = [1.0 / v for v in values]
    
    # Sum of reciprocals (denominator)
    denom = sum(reciprocals)
    
    # Calculate priorities
    # Each priority is the reciprocal divided by the sum of reciprocals
    priorities = [r / denom for r in reciprocals]
    
    return priorities


def apply_macrocategory_scaling(priorities_A, priorities_B, sottofattori, 
                               macro_assignments, scaling_factors):
    """
    CORRECTED IMPLEMENTATION: Apply non-linear scaling using diagonal matrix multiplication
    
    Args:
        priorities_A: Priority vector for category A
        priorities_B: Priority vector for category B  
        sottofattori: List of sub-factors
        macro_assignments: Dict mapping sub-factor to macro-category ('A' or 'B')
        scaling_factors: Dict with scaling factors for each macro-category
    
    Returns:
        Scaled and normalized priority vector
    """
    n_factors = len(sottofattori)
    
    # Create the base priority vector (combining A and B priorities based on assignments)
    base_priorities = []
    for i, sf in enumerate(sottofattori):
        if macro_assignments[sf] == 'A':
            base_priorities.append(priorities_A[i])
        else:  # Category B
            base_priorities.append(priorities_B[i])
    
    # Create diagonal scaling matrix M
    diagonal_matrix = np.zeros((n_factors, n_factors))
    for i, sf in enumerate(sottofattori):
        category = macro_assignments[sf]
        scale_factor = scaling_factors[category]
        diagonal_matrix[i, i] = scale_factor
    
    # Apply matrix multiplication: M * v (where v is the priority vector)
    base_vector = np.array(base_priorities)
    scaled_vector = np.dot(diagonal_matrix, base_vector)
    
    # Normalize the scaled vector
    scaled_list = scaled_vector.tolist()
    return normalize_vector(scaled_list)


def media_geometrica_custom(valori, pesi=None):
    """
    This function calculates the geometric mean:
      - if no weights are provided, use standard geometric mean
      - if weights are provided, use weighted geometric mean
    """
    valori_filtrati = [v for v in valori if v > 0]
    if len(valori_filtrati) == 0:
        return 1.0
    
    if pesi is None:
        # Standard geometric mean
        prodotto = 1.0
        for v in valori_filtrati:
            prodotto *= v
        return prodotto ** (1.0 / len(valori_filtrati))
    else:
        # Weighted geometric mean
        if len(pesi) != len(valori):
            st.error("Weights length does not match values.")
            return 1.0
        
        # Filter weights for positive values only
        pesi_filtrati = [pesi[i] for i, v in enumerate(valori) if v > 0]
        tot = sum(pesi_filtrati)
        
        if tot == 0:
            return 1.0
            
        norm_pesi = [w / tot for w in pesi_filtrati]
        
        prodotto = 1.0
        for v, w in zip(valori_filtrati, norm_pesi):
            prodotto *= v ** w
        return prodotto


def create_bar_chart(labels, values, title="Bar Chart"):
    """Bar-chart plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("DLM", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)


def create_radar_chart(df, title="Radar Chart"):
    """Create a radar plot of the sub-factors to compare different DLMs"""
    categories = list(df.columns)
    N = len(categories)
    
    # Create angles for radar chart
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(df.index)))
    
    for idx, (index, row) in enumerate(df.iterrows()):
        values = row.values.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=index, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    st.pyplot(fig)


def sensitivity_analysis_corrected(df_dlm, sottofattori, priorities_A, priorities_B, 
                                 macro_assignments, base_scaling_factors):
    """
    CORRECTED: Sensitivity analysis with proper macro-category scaling
    """
    st.header("📊 Sensitivity Analysis (Corrected with Macro-category Scaling)")
    
    # Create scaling factor variations for both categories
    scale_A_values = np.linspace(0.1, 2.0, 20)
    scale_B_values = np.linspace(0.1, 2.0, 20)
    
    sensitivity_scores = {dlm: [] for dlm in df_dlm[df_dlm.columns[0]].tolist()}
    
    # Test different combinations of scaling factors
    st.subheader("🔄 Score Variation with Scaling Factor Changes")
    
    # Vary scaling factor A while keeping B constant
    for scale_A in scale_A_values:
        scaling_factors = {
            'A': scale_A,
            'B': base_scaling_factors['B']
        }
        
        # Apply corrected scaling
        final_priorities = apply_macrocategory_scaling(
            priorities_A, priorities_B, sottofattori, 
            macro_assignments, scaling_factors
        )
        
        # Calculate scores for each DLM
        for index, row in df_dlm.iterrows():
            score = sum(float(row[sf]) * final_priorities[i] for i, sf in enumerate(sottofattori))
            dlm_name = row[df_dlm.columns[0]]
            sensitivity_scores[dlm_name].append(score)
    
    # Create DataFrame for sensitivity analysis
    df_sens = pd.DataFrame(sensitivity_scores, index=scale_A_values)
    df_sens.index.name = "Category A Scaling Factor"
    
    # Plot sensitivity analysis
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_sens.columns)))
    for idx, dlm in enumerate(df_sens.columns):
        ax.plot(df_sens.index, df_sens[dlm], marker='o', label=dlm, 
                linewidth=2, markersize=4, color=colors[idx])
    
    ax.set_xlabel("Scaling Factor for Category A", fontsize=12)
    ax.set_ylabel("DLM Score", fontsize=12)
    ax.set_title("Sensitivity of DLM Scores to Category A Scaling Factor", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add vertical line at current scaling factor
    current_scale = base_scaling_factors['A']
    ax.axvline(x=current_scale, color='red', linestyle='--', alpha=0.5, 
               label=f'Current scaling ({current_scale:.2f})')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    return df_sens


def saaty_scale_description():
    """Returns a dictionary of Saaty's scale with complete descriptions"""
    return {
        1: "EQUAL importance",
        2: "Equal to moderate",
        3: "MODERATE importance",
        4: "Moderate to strong",
        5: "STRONG importance",
        6: "Strong to very strong",
        7: "VERY STRONG importance",
        8: "Very strong to extreme",
        9: "EXTREME importance"
    }


def validate_consistency(comparisons):
    """
    Check consistency of comparisons (simplified version for AHP-express)
    Returns True if comparisons are reasonably consistent
    """
    # For AHP-express, we only check that no comparison is 0 or negative
    for values in comparisons.values():
        for v in values:
            if v <= 0:
                return False
    return True


def main():
    st.set_page_config(page_title="AHP-Express Tool Corrected", page_icon="📊", layout="wide")
    
    st.markdown(testo_introduzione, unsafe_allow_html=True)
    st.title("🎯 AHP-Express Tool to Compare DLMs (Corrected Version)")
    
    scala_saaty = saaty_scale_description()
    
    # Step 1: Load DLM file
    st.header("1. 📁 Load DLMs File")
    uploaded_file = st.file_uploader("Upload Excel/CSV file with DLMs", type=['csv', 'xlsx', 'xls'])
    
    if not uploaded_file:
        st.info("Please load a file to proceed. The file should have DLM names in the first column and sub-factors in the remaining columns.")
        st.stop()
    
    df_dlm = carica_dlm_da_file(uploaded_file)
    if df_dlm is None or df_dlm.empty:
        st.error("File not valid or empty")
        st.stop()
    
    st.subheader("Data Preview")
    st.dataframe(df_dlm.head())
    
    # Extract DLM names and sub-factors
    dlm_names = df_dlm[df_dlm.columns[0]].tolist()
    sottofattori = list(df_dlm.columns[1:])
    
    st.success(f"✅ Loaded {len(dlm_names)} DLMs with {len(sottofattori)} sub-factors")
    
    # Step 2: Configure macro-categories assignments and scaling
    st.header("2. ⚙️ Configure Macro-Categories and Scaling Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Macro-Category Assignments")
        st.write("Assign each sub-factor to a macro-category:")
        
        macro_assignments = {}
        for sf in sottofattori:
            macro_assignments[sf] = st.selectbox(
                f"Sub-factor: {sf}",
                options=['A', 'B'],
                key=f"macro_{sf}",
                help=f"Select macro-category for {sf}"
            )
    
    with col2:
        st.subheader("Scaling Factors")
        st.write("Set scaling factors for each macro-category:")
        
        scaling_factors = {}
        scaling_factors['A'] = st.slider(
            "Scaling Factor for Category A",
            min_value=0.1, max_value=2.0, value=1.0, step=0.1,
            help="Higher values increase the importance of Category A sub-factors"
        )
        scaling_factors['B'] = st.slider(
            "Scaling Factor for Category B", 
            min_value=0.1, max_value=2.0, value=1.0, step=0.1,
            help="Higher values increase the importance of Category B sub-factors"
        )
    
    # Display macro-category summary
    st.subheader("📋 Macro-Category Summary")
    category_summary = {}
    for cat in ['A', 'B']:
        factors_in_cat = [sf for sf, assigned_cat in macro_assignments.items() if assigned_cat == cat]
        category_summary[f"Category {cat}"] = {
            "Sub-factors": factors_in_cat,
            "Count": len(factors_in_cat),
            "Scaling Factor": scaling_factors[cat]
        }
    
    for cat_name, info in category_summary.items():
        st.write(f"**{cat_name}** (Scaling: {info['Scaling Factor']:.1f}): {', '.join(info['Sub-factors'])}")
    
    # Step 3: Configure interviews
    st.header("3. 🎤 Configure Interviews")
    num_interviste = st.number_input("Number of Interviews", 
                                     min_value=1, max_value=10, value=1)
    st.info(f"You will need to provide {num_interviste} set(s) of comparisons for each category")
    
    # Step 4: AHP-Express comparisons
    st.header("4. 🔄 AHP-Express Comparisons (Saaty's Scale)")
    
    # Category A comparisons
    st.subheader("📊 Category A Comparisons")
    base_sottofattore_A = st.selectbox("Select reference sub-factor for Category A", 
                                       sottofattori, key="baseA")
    
    comparisons_A = {sf: [] for sf in sottofattori}
    interview_weights_A = []
    
    for i in range(num_interviste):
        with st.expander(f"Interview A #{i + 1}", expanded=True):
            peso_intervista = st.slider(f"Weight of interview A #{i + 1}", 
                                       min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                                       key=f"weightA_{i}")
            interview_weights_A.append(peso_intervista)
            
            st.write(f"**Comparing {base_sottofattore_A} against other sub-factors:**")
            for sf in sottofattori:
                if sf == base_sottofattore_A:
                    comparisons_A[sf].append(1.0)
                    st.write(f"✓ {sf} (reference): 1.0")
                else:
                    val = st.select_slider(
                        f"How much more important is **{base_sottofattore_A}** than **{sf}**?",
                        options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        value=1,
                        key=f"compA_{i}_{sf}",
                        format_func=lambda x: f"{x}: {scala_saaty[x]}"
                    )
                    comparisons_A[sf].append(val)
    
    # Category B comparisons
    st.subheader("📊 Category B Comparisons")
    base_sottofattore_B = st.selectbox("Select reference sub-factor for Category B", 
                                       sottofattori, key="baseB")
    
    comparisons_B = {sf: [] for sf in sottofattori}
    interview_weights_B = []
    
    for i in range(num_interviste):
        with st.expander(f"Interview B #{i + 1}", expanded=True):
            peso_intervista = st.slider(f"Weight of interview B #{i + 1}", 
                                       min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                                       key=f"weightB_{i}")
            interview_weights_B.append(peso_intervista)
            
            st.write(f"**Comparing {base_sottofattore_B} against other sub-factors:**")
            for sf in sottofattori:
                if sf == base_sottofattore_B:
                    comparisons_B[sf].append(1.0)
                    st.write(f"✓ {sf} (reference): 1.0")
                else:
                    val = st.select_slider(
                        f"How much more important is **{base_sottofattore_B}** than **{sf}**?",
                        options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        value=1,
                        key=f"compB_{i}_{sf}",
                        format_func=lambda x: f"{x}: {scala_saaty[x]}"
                    )
                    comparisons_B[sf].append(val)
    
    # Calculate button
    if st.button("🚀 Calculate Priorities and Analysis (Corrected)", type="primary"):
        with st.spinner("Calculating priorities with corrected macro-category scaling..."):
            
            # Validate consistency
            if not validate_consistency(comparisons_A) or not validate_consistency(comparisons_B):
                st.error("Invalid comparisons detected. Please check your inputs.")
                st.stop()
            
            # Category A priority calculation
            st.header("5. 📊 Results and Analysis (Corrected Implementation)")
            
            # Aggregate comparisons using geometric mean
            base_index_A = sottofattori.index(base_sottofattore_A)
            final_ratios_A = [media_geometrica_custom(comparisons_A[sf], interview_weights_A)
                             for sf in sottofattori]
            
            # Calculate priorities using corrected AHP-Express formula
            priA = calculate_ahp_express_prior(base_index_A, final_ratios_A)
            priA_norm = normalize_vector(priA)
            
            # Category B priority calculation
            base_index_B = sottofattori.index(base_sottofattore_B)
            final_ratios_B = [media_geometrica_custom(comparisons_B[sf], interview_weights_B)
                             for sf in sottofattori]
            
            priB = calculate_ahp_express_prior(base_index_B, final_ratios_B)
            priB_norm = normalize_vector(priB)
            
            # CORRECTED: Apply macro-category scaling using diagonal matrix
            final_priorities = apply_macrocategory_scaling(
                priA_norm, priB_norm, sottofattori, macro_assignments, scaling_factors
            )
            
            # Display priority vectors
            st.subheader("📋 Priority Vectors (Corrected)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Category A Priorities**")
                df_priA = pd.DataFrame({
                    "Sub-factor": sottofattori, 
                    "Priority": priA_norm,
                    "Macro-Cat": [macro_assignments[sf] for sf in sottofattori]
                })
                st.dataframe(df_priA.style.format({"Priority": "{:.4f}"}))
            
            with col2:
                st.write("**Category B Priorities**")
                df_priB = pd.DataFrame({
                    "Sub-factor": sottofattori, 
                    "Priority": priB_norm,
                    "Macro-Cat": [macro_assignments[sf] for sf in sottofattori]
                })
                st.dataframe(df_priB.style.format({"Priority": "{:.4f}"}))
            
            with col3:
                st.write("**Final Scaled Priorities**")
                df_final = pd.DataFrame({
                    "Sub-factor": sottofattori, 
                    "Scaled Priority": final_priorities,
                    "Scaling Factor": [scaling_factors[macro_assignments[sf]] for sf in sottofattori]
                })
                st.dataframe(df_final.style.format({
                    "Scaled Priority": "{:.4f}",
                    "Scaling Factor": "{:.1f}"
                }))
            
            # Calculate DLM scores
            data_scores = []
            for _, row in df_dlm.iterrows():
                score = sum(float(row[sf]) * final_priorities[i] 
                           for i, sf in enumerate(sottofattori))
                data_scores.append(score)
            
            df_dlm['Final_Score'] = data_scores
            
            # Display rankings
            st.subheader("🏆 DLM Rankings (with Corrected Scaling)")
            data_sorted = df_dlm.sort_values(by="Final_Score", ascending=False).copy()
            data_sorted['Rank'] = range(1, len(data_sorted) + 1)
            
            # Format the display
            display_cols = ['Rank', df_dlm.columns[0], 'Final_Score'] + sottofattori
            st.dataframe(
                data_sorted[display_cols].style.format({
                    'Final_Score': '{:.4f}',
                    **{sf: '{:.2f}' for sf in sottofattori}
                }).background_gradient(subset=['Final_Score'], cmap='viridis')
            )
            
            # Winner announcement
            winner = data_sorted.iloc[0]
            st.success(f"🥇 **Best DLM:** {winner[df_dlm.columns[0]]} with score {winner['Final_Score']:.4f}")
            
            # Visualizations
            st.subheader("📊 Visualizations")
            
            tab1, tab2, tab3 = st.tabs(["Bar Chart", "Radar Chart", "Priority Distribution"])
            
            with tab1:
                create_bar_chart(
                    data_sorted[df_dlm.columns[0]].tolist(), 
                    data_sorted["Final_Score"].tolist(),
                    title="DLM Final Scores Comparison (Corrected Scaling)"
                )
            
            with tab2:
                # Prepare radar chart data
                df_radar = pd.DataFrame(index=df_dlm[df_dlm.columns[0]], columns=sottofattori)
                for _, row in df_dlm.iterrows():
                    nome = row[df_dlm.columns[0]]
                    for i, sf in enumerate(sottofattori):
                        try:
                            val = float(row[sf])
                        except:
                            val = 0.0
                        df_radar.loc[nome, sf] = val
                
                # Normalize each sub-factor for better visualization
                for sf in sottofattori:
                    col = df_radar[sf].astype(float)
                    max_val = col.max()
                    if max_val > 0:
                        df_radar[sf] = col / max_val
                    else:
                        df_radar[sf] = 0.0
                
                create_radar_chart(df_radar, title="DLM Comparison across Sub-factors (Normalized)")
            
            with tab3:
                # Show priority distribution
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Pie chart for final priorities with macro-category colors
                colors_A = plt.cm.Blues(np.linspace(0.3, 0.8, sum(1 for sf in sottofattori if macro_assignments[sf] == 'A')))
                colors_B = plt.cm.Reds(np.linspace(0.3, 0.8, sum(1 for sf in sottofattori if macro_assignments[sf] == 'B')))
                
                pie_colors = []
                color_A_idx, color_B_idx = 0, 0
                for sf in sottofattori:
                    if macro_assignments[sf] == 'A':
                        pie_colors.append(colors_A[color_A_idx])
                        color_A_idx += 1
                    else:
                        pie_colors.append(colors_B[color_B_idx])
                        color_B_idx += 1
                
                ax1.pie(final_priorities, labels=sottofattori, autopct='%1.1f%%', 
                       colors=pie_colors, startangle=90)
                ax1.set_title("Final Scaled Priority Distribution", fontsize=12, fontweight='bold')
                
                # Comparison of priorities with scaling factors
                x = np.arange(len(sottofattori))
                width = 0.25
                
                ax2.bar(x - width, priA_norm, width, label='Category A', alpha=0.8)
                ax2.bar(x, priB_norm, width, label='Category B', alpha=0.8)
                ax2.bar(x + width, final_priorities, width, label='Final Scaled', alpha=0.8)
                ax2.set_xlabel('Sub-factors')
                ax2.set_ylabel('Priority')
                ax2.set_title('Priority Comparison: Before & After Scaling', fontsize=12, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(sottofattori, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Corrected Sensitivity Analysis
            sensitivity_analysis_corrected(df_dlm, sottofattori, priA_norm, priB_norm, 
                                          macro_assignments, scaling_factors)
            
            # Export results
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Create Excel file for download
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    data_sorted.to_excel(writer, sheet_name='Rankings', index=False)
                    df_final.to_excel(writer, sheet_name='Final_Priorities', index=False)
                    df_priA.to_excel(writer, sheet_name='Category_A', index=False)
                    df_priB.to_excel(writer, sheet_name='Category_B', index=False)
                    
                    # Add macro-category assignments
                    df_assignments = pd.DataFrame({
                        'Sub-factor': sottofattori,
                        'Macro-Category': [macro_assignments[sf] for sf in sottofattori],
                        'Scaling Factor': [scaling_factors[macro_assignments[sf]] for sf in sottofattori]
                    })
                    df_assignments.to_excel(writer, sheet_name='Macro_Categories', index=False)
                
                output.seek(0)
                st.download_button(
                    label="📥 Download Results (Excel)",
                    data=output,
                    file_name="ahp_express_corrected_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # Create CSV for download
                csv = data_sorted.to_csv(index=False)
                st.download_button(
                    label="📥 Download Rankings (CSV)",
                    data=csv,
                    file_name="dlm_rankings_corrected.csv",
                    mime="text/csv"
                )
            
            # Corrected Methodology note
            with st.expander("📚 Corrected Methodology Notes"):
                st.markdown("""
                ### AHP-Express with Corrected Macro-Category Scaling
                
                This corrected tool implements the **AHP-Express** method with proper 
                non-linear macro-category scaling using diagonal matrix multiplication.
                
                **Key corrections implemented:**
                
                1. **Macro-category Assignment**: Each sub-factor is explicitly assigned to a macro-category
                2. **Diagonal Matrix Scaling**: Instead of linear weighted combination, we use:
                   - Create diagonal matrix M where M[i,i] = scaling_factor[macro_category[i]]
                   - Apply scaling: scaled_vector = M × priority_vector
                   - Then normalize: final_priorities = scaled_vector / sum(scaled_vector)
                
                **Priority calculation formula:**
                - For sub-factor i in macro-category k with scaling factor s_k:
                - Scaled_Priority(i) = Original_Priority(i) × s_k
                - Final_Priority(i) = Scaled_Priority(i) / Σ(Scaled_Priority(j)) for all j
                
                **Mathematical representation:**
                ```
                If v = [v0, v1, ..., vN-1] is the priority vector
                And M is the diagonal matrix with scaling factors
                Then: final_priorities = normalize(M × v)
                ```
                
                **Benefits of this approach:**
                - Non-linear scaling preserves the relative importance within categories
                - Allows fine-tuning of macro-category influence
                - Maintains mathematical consistency with AHP principles
                - Provides more control over decision-making process
                """)


if __name__ == "__main__":
    main()
