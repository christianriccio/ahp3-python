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

2. **Macro-categories with Weights and Scaling**  
   - Sub-factors can optionally be assigned to macro-categories (A, B, or None/Neutral)
   - Each macro-category has two parameters:
     - **Weight**: How important is this category in the overall decision (peso_A, peso_B)
     - **Scaling Factor**: How to amplify/dampen the priorities within this category
   - Neutral sub-factors are not affected by macro-category weights or scaling

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
   - The final priority is calculated as:
     - For category A factors: priority_A * peso_A * scaling_factor_A
     - For category B factors: priority_B * peso_B * scaling_factor_B  
     - For neutral factors: priority_neutral * neutral_weight
   - Then normalized and used to score the DLMs.

**Getting Started**:  
- Upload an *Excel/CSV* file containing the DLMs and their sub-factors.  
- Optionally assign sub-factors to macro-categories and set weights/scaling factors.
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


def calculate_final_priorities_corrected(priorities_A, priorities_B, sottofattori,
                                       macro_assignments, category_weights, scaling_factors):
    """
    CORRECTED LOGIC: 
    1. FIRST combine priorities with category weights (original formula)
    2. THEN apply scaling factors based on category assignments (correction)
    3. Normalize the result
    """
    # Step 1: Apply ORIGINAL formula to combine priorities with weights
    combined_priorities = []
    for i in range(len(priorities_A)):
        combined_priority = (priorities_A[i] * category_weights['A'] + 
                           priorities_B[i] * category_weights['B'])
        combined_priorities.append(combined_priority)
    
    # Step 2: Apply scaling factors using diagonal matrix multiplication
    n_factors = len(sottofattori)
    diagonal_matrix = np.zeros((n_factors, n_factors))
    
    for i, sf in enumerate(sottofattori):
        category = macro_assignments.get(sf, 'None')
        if category == 'A':
            diagonal_matrix[i, i] = scaling_factors['A']
        elif category == 'B':
            diagonal_matrix[i, i] = scaling_factors['B']
        else:
            diagonal_matrix[i, i] = 1.0  # No scaling for unassigned factors
    
    # Apply diagonal matrix multiplication
    combined_array = np.array(combined_priorities)
    scaled_array = np.dot(diagonal_matrix, combined_array)
    
    # Step 3: Normalize the final result
    return normalize_vector(scaled_array.tolist())


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
                                 macro_assignments, base_category_weights, base_scaling_factors):
    """
    CORRECTED: Sensitivity analysis with proper sequence: combine weights first, then scale
    """
    st.header("📊 Sensitivity Analysis")
    
    # Create variations for category weights
    peso_A_values = np.linspace(0, 1, 21)
    sensitivity_scores = {dlm: [] for dlm in df_dlm[df_dlm.columns[0]].tolist()}
    
    # Test different category weight combinations
    for peso_A in peso_A_values:
        peso_B = 1 - peso_A
        category_weights = {'A': peso_A, 'B': peso_B}
        
        # Calculate final priorities with current weights using corrected logic
        final_priorities = calculate_final_priorities_corrected(
            priorities_A, priorities_B, sottofattori,
            macro_assignments, category_weights, base_scaling_factors
        )
        
        # Calculate scores for each DLM
        for index, row in df_dlm.iterrows():
            score = sum(float(row[sf]) * final_priorities[i] for i, sf in enumerate(sottofattori))
            dlm_name = row[df_dlm.columns[0]]
            sensitivity_scores[dlm_name].append(score)
    
    # Create DataFrame for sensitivity analysis
    df_sens = pd.DataFrame(sensitivity_scores, index=peso_A_values)
    df_sens.index.name = "Category A Weight"
    
    # Plot sensitivity analysis
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_sens.columns)))
    for idx, dlm in enumerate(df_sens.columns):
        ax.plot(df_sens.index, df_sens[dlm], marker='o', label=dlm, 
                linewidth=2, markersize=4, color=colors[idx])
    
    ax.set_xlabel("Weight of Category A", fontsize=12)
    ax.set_ylabel("DLM Score", fontsize=12)
    ax.set_title("Sensitivity of DLM Scores to Category A Weight", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Add vertical line at current weight
    current_weight = base_category_weights['A']
    ax.axvline(x=current_weight, color='red', linestyle='--', alpha=0.5, 
               label=f'Current weight ({current_weight:.2f})')
    
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
    
    # Step 2: Configure macro-categories (optional)
    st.header("2. ⚙️ Configure Macro-Categories (Optional)")
    
    # Enable macro-category system
    use_macro_categories = st.checkbox("Enable Macro-Category System", value=True,
                                      help="If enabled, you can assign sub-factors to categories A/B and set weights/scaling")
    
    if use_macro_categories:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Category Assignments")
            st.write("Assign sub-factors to categories (optional):")
            
            macro_assignments = {}
            for sf in sottofattori:
                macro_assignments[sf] = st.selectbox(
                    f"{sf}",
                    options=['None', 'A', 'B'],
                    key=f"macro_{sf}",
                    help=f"Select category for {sf} (None = neutral)"
                )
        
        with col2:
            st.subheader("Category Weights")
            st.write("Set importance weights for each category:")
            
            peso_A = st.slider(
                "Weight for Category A",
                min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                help="How important is Category A in the final decision"
            )
            peso_B = 1 - peso_A
            st.write(f"Weight for Category B: {peso_B:.2f}")
            
            category_weights = {'A': peso_A, 'B': peso_B}
        
        with col3:
            st.subheader("Scaling Factors")
            st.write("Set scaling multipliers for each category:")
            
            scaling_A = st.slider(
                "Scaling Factor for Category A",
                min_value=0.1, max_value=3.0, value=1.0, step=0.1,
                help="Multiplier applied to sub-factors assigned to Category A"
            )
            scaling_B = st.slider(
                "Scaling Factor for Category B", 
                min_value=0.1, max_value=3.0, value=1.0, step=0.1,
                help="Multiplier applied to sub-factors assigned to Category B"
            )
            
            scaling_factors = {'A': scaling_A, 'B': scaling_B}
        
        # Display category summary
        st.subheader("📋 Category Configuration Summary")
        
        categories_summary = {
            'A': [sf for sf, cat in macro_assignments.items() if cat == 'A'],
            'B': [sf for sf, cat in macro_assignments.items() if cat == 'B'],
            'Neutral': [sf for sf, cat in macro_assignments.items() if cat == 'None']
        }
        
        for cat_name, factors in categories_summary.items():
            if factors:
                if cat_name in ['A', 'B']:
                    weight = category_weights[cat_name]
                    scaling = scaling_factors[cat_name]
                    st.write(f"**Category {cat_name}** (Weight: {weight:.2f}, Scaling: {scaling:.1f}): {', '.join(factors)}")
                else:
                    st.write(f"**{cat_name}** (No specific scaling): {', '.join(factors)}")
    
    else:
        # Default: no macro-categories, all factors treated equally
        macro_assignments = {sf: 'None' for sf in sottofattori}
        category_weights = {'A': 0.5, 'B': 0.5}
        scaling_factors = {'A': 1.0, 'B': 1.0}
        st.info("Macro-category system disabled. All sub-factors will be treated equally.")
    
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
    if st.button("🚀 Calculate Priorities and Analysis", type="primary"):
        with st.spinner("Calculating priorities..."):
            
            # Validate consistency
            if not validate_consistency(comparisons_A) or not validate_consistency(comparisons_B):
                st.error("Invalid comparisons detected. Please check your inputs.")
                st.stop()
            
            # Calculate priorities for each category
            st.header("5. 📊 Results and Analysis")
            
            # Category A priority calculation
            base_index_A = sottofattori.index(base_sottofattore_A)
            final_ratios_A = [media_geometrica_custom(comparisons_A[sf], interview_weights_A)
                             for sf in sottofattori]
            
            priA = calculate_ahp_express_prior(base_index_A, final_ratios_A)
            priA_norm = normalize_vector(priA)
            
            # Category B priority calculation
            base_index_B = sottofattori.index(base_sottofattore_B)
            final_ratios_B = [media_geometrica_custom(comparisons_B[sf], interview_weights_B)
                             for sf in sottofattori]
            
            priB = calculate_ahp_express_prior(base_index_B, final_ratios_B)
            priB_norm = normalize_vector(priB)
            
            # CORRECTED: Apply the correct sequence - FIRST combine with weights, THEN scale
            final_priorities = calculate_final_priorities_corrected(
                priA_norm, priB_norm, sottofattori,
                macro_assignments, category_weights, scaling_factors
            )
            
            # Display priority vectors
            st.subheader("📋 Priority Vectors")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Category A Priorities**")
                df_priA = pd.DataFrame({
                    "Sub-factor": sottofattori, 
                    "Priority": priA_norm,
                    "Assignment": [macro_assignments[sf] for sf in sottofattori]
                })
                st.dataframe(df_priA.style.format({"Priority": "{:.4f}"}))
            
            with col2:
                st.write("**Category B Priorities**")
                df_priB = pd.DataFrame({
                    "Sub-factor": sottofattori, 
                    "Priority": priB_norm,
                    "Assignment": [macro_assignments[sf] for sf in sottofattori]
                })
                st.dataframe(df_priB.style.format({"Priority": "{:.4f}"}))
            
            with col3:
                st.write("**Final Combined Priorities**")
                
                # Show the effect of scaling
                scaled_display = []
                for i, sf in enumerate(sottofattori):
                    category = macro_assignments[sf]
                    if category == 'A':
                        scaling_applied = scaling_factors['A']
                    elif category == 'B':
                        scaling_applied = scaling_factors['B']
                    else:
                        scaling_applied = 1.0
                    scaled_display.append(scaling_applied)
                
                df_final = pd.DataFrame({
                    "Sub-factor": sottofattori, 
                    "Final Priority": final_priorities,
                    "Category": [macro_assignments[sf] for sf in sottofattori],
                    "Scaling Applied": scaled_display
                })
                st.dataframe(df_final.style.format({
                    "Final Priority": "{:.4f}",
                    "Scaling Applied": "{:.1f}"
                }))
            
            # Calculate DLM scores
            data_scores = []
            for _, row in df_dlm.iterrows():
                score = sum(float(row[sf]) * final_priorities[i] 
                           for i, sf in enumerate(sottofattori))
                data_scores.append(score)
            
            df_dlm['Final_Score'] = data_scores
            
            # Display rankings
            st.subheader("🏆 DLM Rankings")
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
            
            tab1, tab2, tab3 = st.tabs(["Bar Chart", "Radar Chart", "Priority Analysis"])
            
            with tab1:
                create_bar_chart(
                    data_sorted[df_dlm.columns[0]].tolist(), 
                    data_sorted["Final_Score"].tolist(),
                    title="DLM Final Scores Comparison"
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
                # Show priority analysis
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                # Priority distribution by category
                category_colors = {'A': 'lightblue', 'B': 'lightcoral', 'None': 'lightgray'}
                colors_by_cat = [category_colors[macro_assignments[sf]] for sf in sottofattori]
                
                ax1.pie(final_priorities, labels=sottofattori, autopct='%1.1f%%', 
                       colors=colors_by_cat, startangle=90)
                ax1.set_title("Final Priority Distribution by Category", fontsize=12, fontweight='bold')
                
                # Comparison between categories
                x = np.arange(len(sottofattori))
                width = 0.25
                
                ax2.bar(x - width, priA_norm, width, label='Category A', alpha=0.8)
                ax2.bar(x, priB_norm, width, label='Category B', alpha=0.8)
                ax2.bar(x + width, final_priorities, width, label='Final Combined', alpha=0.8)
                ax2.set_xlabel('Sub-factors')
                ax2.set_ylabel('Priority')
                ax2.set_title('Priority Comparison: A vs B vs Final', fontsize=12, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(sottofattori, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(axis='y', alpha=0.3)
                
                # Effect of weights and scaling
                effect_analysis = []
                for i, sf in enumerate(sottofattori):
                    cat = macro_assignments[sf]
                    if cat == 'A':
                        base_priority = priA_norm[i]
                        weight = category_weights['A']
                        scaling = scaling_factors['A']
                    elif cat == 'B':
                        base_priority = priB_norm[i] 
                        weight = category_weights['B']
                        scaling = scaling_factors['B']
                    else:
                        base_priority = (priA_norm[i] + priB_norm[i]) / 2
                        weight = neutral_weight
                        scaling = 1.0
                    
                    effect_analysis.append({
                        'factor': sf,
                        'base': base_priority,
                        'after_weight': base_priority * weight,
                        'final': final_priorities[i]
                    })
                
                factors = [e['factor'] for e in effect_analysis]
                base_vals = [e['base'] for e in effect_analysis]
                weighted_vals = [e['after_weight'] for e in effect_analysis]
                final_vals = [e['final'] for e in effect_analysis]
                
                x = np.arange(len(factors))
                ax3.bar(x - 0.2, base_vals, 0.2, label='Base Priority', alpha=0.8)
                ax3.bar(x, weighted_vals, 0.2, label='After Weight', alpha=0.8)
                ax3.bar(x + 0.2, final_vals, 0.2, label='After Scaling', alpha=0.8)
                ax3.set_xlabel('Sub-factors')
                ax3.set_ylabel('Priority Value')
                ax3.set_title('Effect of Weights and Scaling', fontsize=12, fontweight='bold')
                ax3.set_xticks(x)
                ax3.set_xticklabels(factors, rotation=45, ha='right')
                ax3.legend()
                ax3.grid(axis='y', alpha=0.3)
                
                # Category summary
                cat_summary = {'A': [], 'B': [], 'Neutral': []}
                for i, sf in enumerate(sottofattori):
                    cat = macro_assignments[sf]
                    cat_key = 'Neutral' if cat == 'None' else cat
                    cat_summary[cat_key].append(final_priorities[i])
                
                cat_totals = {k: sum(v) for k, v in cat_summary.items() if v}
                if cat_totals:
                    ax4.bar(cat_totals.keys(), cat_totals.values(), 
                           color=['lightblue', 'lightcoral', 'lightgray'][:len(cat_totals)])
                    ax4.set_title('Total Priority by Category', fontsize=12, fontweight='bold')
                    ax4.set_ylabel('Total Priority')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Sensitivity Analysis
            if use_macro_categories:
                sensitivity_analysis_corrected(df_dlm, sottofattori, priA_norm, priB_norm, 
                                              macro_assignments, category_weights, scaling_factors)
            
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
                
                output.seek(0)
                st.download_button(
                    label="📥 Download Results (Excel)",
                    data=output,
                    file_name="ahp_express_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col2:
                # Create CSV for download
                csv = data_sorted.to_csv(index=False)
                st.download_button(
                    label="📥 Download Rankings (CSV)",
                    data=csv,
                    file_name="dlm_rankings.csv",
                    mime="text/csv"
                )
            
            # Methodology note
            with st.expander("📚 Corrected Methodology Notes"):
                st.markdown("""
                ### CORRECTED AHP-Express with Macro-Category Scaling
                
                **The corrected implementation follows this exact sequence:**
                
                **Step 1: Calculate Category Priorities (AHP-Express)**
                - Category A priorities: priA_norm (from AHP-Express)
                - Category B priorities: priB_norm (from AHP-Express)
                
                **Step 2: Combine with Category Weights (ORIGINAL FORMULA)**
                ```
                combined_priorities[i] = priA_norm[i] × peso_A + priB_norm[i] × peso_B
                ```
                This is your ORIGINAL formula - here the category weights DO matter!
                
                **Step 3: Apply Scaling Factors (CORRECTION via Diagonal Matrix)**
                - Create diagonal matrix M where M[i,i] = scaling_factor for the category of sub-factor i
                - Apply: scaled_priorities = M × combined_priorities
                
                **Step 4: Normalize**
                ```
                final_priorities = normalize(scaled_priorities)
                ```
                
                **Why this was wrong before:**
                I was applying scaling to priA and priB SEPARATELY, which completely bypassed 
                the category weight combination. The category weights (peso_A, peso_B) had no 
                effect because they were applied to already-scaled vectors.
                
                **Why this is correct now:**
                1. Category weights control the RELATIVE IMPORTANCE of categories A vs B
                2. Scaling factors control the AMPLIFICATION of sub-factors within their assigned category
                3. The two effects are applied in the correct mathematical sequence
                
                **Mathematical representation:**
                ```
                Step 1: combined[i] = priA[i] × peso_A + priB[i] × peso_B
                Step 2: M = diag[s_cat[0], s_cat[1], ..., s_cat[n-1]]  # scaling by category assignment
                Step 3: final = normalize(M × combined)
                ```
                
                **Now when you change:**
                - **peso_A/peso_B**: Changes the relative importance of categories → affects ranking
                - **scaling_A/scaling_B**: Amplifies/dampens sub-factors in each category → affects ranking
                - **Category assignments**: Changes which scaling factor applies to each sub-factor
                """)
                
                # Show current parameter values
                st.subheader("Current Configuration")
                st.write(f"**Category Weights:** A={category_weights['A']:.2f}, B={category_weights['B']:.2f}")
                st.write(f"**Scaling Factors:** A={scaling_factors['A']:.1f}, B={scaling_factors['B']:.1f}")
                
                assigned_factors = {
                    'A': [sf for sf, cat in macro_assignments.items() if cat == 'A'],
                    'B': [sf for sf, cat in macro_assignments.items() if cat == 'B'], 
                    'None': [sf for sf, cat in macro_assignments.items() if cat == 'None']
                }
                
                for cat, factors in assigned_factors.items():
                    if factors:
                        st.write(f"**Category {cat}:** {', '.join(factors)}")
                
                # Show what should happen when parameters change
                st.subheader("Expected Effects of Parameter Changes")
                st.write("✅ **Changing peso_A/peso_B should**: Shift ranking towards factors favored by A vs B priorities")
                st.write("✅ **Changing scaling_A/scaling_B should**: Amplify/dampen factors assigned to that category")
                st.write("✅ **Assigning factors to categories should**: Apply the respective scaling factor to those factors")


if __name__ == "__main__":
    main()
