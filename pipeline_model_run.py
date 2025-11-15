"""
Political Uncertainty & Corporate Investment Analysis
Panel Econometrics + Double Machine Learning for Causal Inference
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_process_epu_data(filepath):
    """Load and process Economic Policy Uncertainty (EPU) data."""
    print("\n📊 Processing Political Uncertainty Data...")
    
    us_epu = pd.read_excel(filepath, skipfooter=10)
    us_epu = us_epu[pd.to_numeric(us_epu['Year'], errors='coerce').notna() & 
                    pd.to_numeric(us_epu['Month'], errors='coerce').notna()]
    
    us_epu['Year'] = pd.to_numeric(us_epu['Year'])
    us_epu['Month'] = pd.to_numeric(us_epu['Month'])
    us_epu['News_Based_Policy_Uncert_Index'] = pd.to_numeric(
        us_epu['News_Based_Policy_Uncert_Index'], errors='coerce'
    )
    us_epu = us_epu.dropna(subset=['Year', 'Month', 'News_Based_Policy_Uncert_Index'])
    
    us_epu['date'] = pd.to_datetime(us_epu[['Year', 'Month']].assign(day=1))
    us_epu['year_quarter'] = us_epu['date'].dt.to_period('Q')
    
    epu_quarterly = us_epu.groupby('year_quarter', as_index=False)['News_Based_Policy_Uncert_Index'].mean()
    epu_quarterly['date'] = epu_quarterly['year_quarter'].dt.end_time
    
    print(f"✅ EPU data processed: {epu_quarterly.shape[0]} quarters")
    return epu_quarterly


def load_and_merge_financial_data(balance_path, cashflow_path, income_path):
    """Load and merge financial statement data."""
    print("\n🏢 Processing Firm Financial Data...")
    
    balance_q = pd.read_csv(balance_path, usecols=['SimFinId', 'Publish Date', 'Total Assets', 'Total Liabilities'])
    cashflow_q = pd.read_csv(cashflow_path, usecols=['SimFinId', 'Publish Date', 
                                                      'Net Cash from Operating Activities',
                                                      'Net Cash from Investing Activities'])
    income_q = pd.read_csv(income_path, usecols=['SimFinId', 'Publish Date', 'Net Income'])
    
    for df in [balance_q, cashflow_q, income_q]:
        df['report_date'] = pd.to_datetime(df['Publish Date'])
        df['year_quarter'] = df['report_date'].dt.to_period('Q')
        df.drop('Publish Date', axis=1, inplace=True)
    
    financial_data = balance_q.merge(cashflow_q, on=['SimFinId', 'year_quarter'], how='inner')\
                              .merge(income_q, on=['SimFinId', 'year_quarter'], how='inner')
    
    print(f"✅ Financial data merged: {financial_data.shape[0]} observations")
    return financial_data


def create_analysis_variables(financial_data):
    """Create key financial ratios and lagged variables."""
    print("\n🔧 Creating Analysis Variables...")
    
    financial_data = financial_data.sort_values(['SimFinId', 'year_quarter'])
    
    financial_data['investment_ratio'] = abs(financial_data['Net Cash from Investing Activities']) / financial_data['Total Assets']
    financial_data['cash_flow_ratio'] = financial_data['Net Cash from Operating Activities'] / financial_data['Total Assets']
    financial_data['leverage'] = financial_data['Total Liabilities'] / financial_data['Total Assets']
    financial_data['size'] = np.log(financial_data['Total Assets'])
    financial_data['profitability'] = financial_data['Net Income'] / financial_data['Total Assets']
    
    financial_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    ratio_cols = ['investment_ratio', 'cash_flow_ratio', 'leverage', 'profitability']
    financial_data.dropna(subset=ratio_cols + ['size'], inplace=True)
    
    for col in ratio_cols:
        lower, upper = financial_data[col].quantile([0.01, 0.99])
        financial_data = financial_data[(financial_data[col] >= lower) & (financial_data[col] <= upper)]
    
    lag_cols = ratio_cols + ['size']
    for col in lag_cols:
        financial_data[f'lag1_{col}'] = financial_data.groupby('SimFinId')[col].shift(1)
    
    financial_data.dropna(subset=['lag1_investment_ratio'], inplace=True)
    
    print(f"✅ Analysis variables created: {financial_data.shape[0]} observations")
    return financial_data


def merge_epu_and_financial(epu_quarterly, financial_data):
    """Merge EPU data with financial data and create treatment variable."""
    print("\n🔗 Merging EPU with Financial Data...")
    
    financial_data['year_quarter'] = financial_data['year_quarter'].astype(str)
    epu_quarterly['year_quarter'] = epu_quarterly['year_quarter'].astype(str)
    
    min_year = financial_data['report_date'].min().year
    epu_filtered = epu_quarterly[epu_quarterly['date'].dt.year >= min_year].copy()
    
    final_df = financial_data.merge(epu_filtered[['year_quarter', 'News_Based_Policy_Uncert_Index']],
                                    on='year_quarter', how='inner')
    
    epu_threshold = final_df['News_Based_Policy_Uncert_Index'].quantile(0.75)
    final_df['high_uncertainty'] = (final_df['News_Based_Policy_Uncert_Index'] > epu_threshold).astype(int)
    
    print(f"✅ Final dataset: {final_df.shape[0]} observations")
    print(f"📈 Treatment prevalence: {final_df['high_uncertainty'].mean():.2%}")
    
    return final_df


def exploratory_analysis(final_df, epu_quarterly):
    """Perform comprehensive exploratory analysis."""
    print("\n📊 Exploratory Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    low_unc = final_df[final_df['high_uncertainty'] == 0]['investment_ratio']
    high_unc = final_df[final_df['high_uncertainty'] == 1]['investment_ratio']
    
    # Plot 1: Investment distribution
    axes[0,0].hist([low_unc, high_unc], bins=30, alpha=0.7,
                   label=['Low Uncertainty', 'High Uncertainty'],
                   color=['lightblue', 'lightcoral'], edgecolor='black')
    axes[0,0].set_title('Investment Ratio Distribution by Uncertainty Regime', fontweight='bold')
    axes[0,0].set_xlabel('Investment Ratio')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: EPU over time
    high_unc_periods = final_df.groupby('year_quarter')['high_uncertainty'].max()
    epu_time = epu_quarterly.set_index('year_quarter').reindex(high_unc_periods.index)
    
    axes[0,1].plot(epu_time.index, epu_time['News_Based_Policy_Uncert_Index'],
                   linewidth=2, label='EPU Index', color='darkred')
    axes[0,1].fill_between(epu_time.index, 0, epu_time['News_Based_Policy_Uncert_Index'],
                          where=(high_unc_periods == 1), alpha=0.3, color='red', 
                          label='High Uncertainty Periods')
    axes[0,1].set_title('EPU Index with High-Uncertainty Periods', fontweight='bold')
    axes[0,1].set_ylabel('EPU Index')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Box plot
    box_plot = axes[1,0].boxplot([low_unc, high_unc], 
                                 labels=['Low Uncertainty', 'High Uncertainty'],
                                 patch_artist=True)
    for patch, color in zip(box_plot['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)
    
    axes[1,0].set_title('Investment Ratio: Low vs High Uncertainty', fontweight='bold')
    axes[1,0].set_ylabel('Investment Ratio')
    axes[1,0].grid(True, alpha=0.3)
    
    for i, (data, label) in enumerate([(low_unc, 'Low'), (high_unc, 'High')], 1):
        axes[1,0].text(i, data.mean() + 0.002, f'Mean: {data.mean():.4f}',
                      ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Correlation heatmap
    corr_vars = ['investment_ratio', 'cash_flow_ratio', 'leverage', 'size',
                 'profitability', 'News_Based_Policy_Uncert_Index', 'high_uncertainty']
    corr_matrix = final_df[corr_vars].corr()
    
    im = axes[1,1].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[1,1].set_xticks(range(len(corr_vars)))
    axes[1,1].set_yticks(range(len(corr_vars)))
    axes[1,1].set_xticklabels(corr_vars, rotation=45, ha='right')
    axes[1,1].set_yticklabels(corr_vars)
    axes[1,1].set_title('Correlation Matrix', fontweight='bold')
    
    for i in range(len(corr_vars)):
        for j in range(len(corr_vars)):
            axes[1,1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', fontsize=9, fontweight='bold',
                          color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=axes[1,1], shrink=0.6)
    plt.tight_layout()
    plt.savefig('plots/exploratory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(low_unc, high_unc, equal_var=False)
    
    print(f"\n📊 Statistical Test Results:")
    print(f"Mean investment (Low Uncertainty): {low_unc.mean():.4f}")
    print(f"Mean investment (High Uncertainty): {high_unc.mean():.4f}")
    print(f"Difference: {low_unc.mean() - high_unc.mean():.4f}")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    print("✅ Statistically significant!" if p_value < 0.05 else "❌ Not significant")


# ============================================================================
# PANEL ECONOMETRICS
# ============================================================================

def run_panel_econometrics(final_df):
    """Run benchmark panel econometric models."""
    print("\n📊 Running Panel Econometrics...")
    
    df = final_df.copy()
    
    # Model 1: Pooled OLS
    print("  Model 1: Pooled OLS...")
    m1 = smf.ols('investment_ratio ~ high_uncertainty', data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['SimFinId']}
    )
    
    # Model 2: Firm FE
    print("  Model 2: Firm Fixed Effects...")
    df['inv_fe'] = df.groupby('SimFinId')['investment_ratio'].transform(lambda x: x - x.mean())
    df['unc_fe'] = df.groupby('SimFinId')['high_uncertainty'].transform(lambda x: x - x.mean())
    m2 = smf.ols('inv_fe ~ unc_fe - 1', data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['SimFinId']}
    )
    
    # Model 3: Time FE
    print("  Model 3: Time Fixed Effects...")
    m3 = smf.ols('investment_ratio ~ high_uncertainty + C(year_quarter)', data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['SimFinId']}
    )
    
    # Model 4: Two-way FE with controls
    print("  Model 4: Two-way FE + Controls...")
    has_variation = df.groupby('SimFinId')['high_uncertainty'].std() > 0
    df_var = df[df['SimFinId'].isin(has_variation[has_variation].index)].copy()
    
    controls = ['lag1_cash_flow_ratio', 'lag1_leverage', 'lag1_size', 'lag1_profitability']
    
    for var in ['investment_ratio', 'high_uncertainty'] + controls:
        firm_mean = df_var.groupby('SimFinId')[var].transform('mean')
        time_mean = df_var.groupby('year_quarter')[var].transform('mean')
        grand_mean = df_var[var].mean()
        df_var[f'{var}_tw'] = df_var[var] - firm_mean - time_mean + grand_mean
    
    formula = 'investment_ratio_tw ~ high_uncertainty_tw + ' + ' + '.join([f'{c}_tw' for c in controls]) + ' - 1'
    m4 = smf.ols(formula, data=df_var).fit(
        cov_type='cluster', cov_kwds={'groups': df_var['SimFinId']}
    )
    
    print("✅ Panel econometrics complete!")
    
    return {'pooled': m1, 'firm_fe': m2, 'time_fe': m3, 'twoway_fe': m4}


# ============================================================================
# DOUBLE MACHINE LEARNING
# ============================================================================

def double_ml_estimation(df, outcome='investment_ratio', treatment='high_uncertainty', 
                         controls=None, n_folds=5):
    """
    Double/Debiased Machine Learning for causal inference.
    
    Uses cross-fitting to estimate:
    1. E[Y|X] - outcome model
    2. E[D|X] - treatment model
    Then estimates ATE from residuals.
    """
    print("\n🤖 Running Double Machine Learning...")
    
    if controls is None:
        controls = ['lag1_cash_flow_ratio', 'lag1_leverage', 'lag1_size', 
                   'lag1_profitability', 'News_Based_Policy_Uncert_Index']
    
    data = df[[outcome, treatment] + controls].dropna().copy()
    Y = data[outcome].values
    D = data[treatment].values
    X = data[controls].values
    
    print(f"  Sample size: {len(data)}")
    print(f"  Controls: {len(controls)}")
    print(f"  Cross-fitting with {n_folds} folds...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    Y_res = np.zeros(len(Y))
    D_res = np.zeros(len(D))
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
        print(f"    Fold {fold}/{n_folds}", end='\r')
        
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        D_train, D_test = D[train_idx], D[test_idx]
        
        # E[Y|X] using Random Forest
        m_Y = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                     min_samples_leaf=20, random_state=42, n_jobs=-1)
        m_Y.fit(X_train, Y_train)
        Y_res[test_idx] = Y_test - m_Y.predict(X_test)
        
        # E[D|X] using Gradient Boosting
        m_D = GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                        learning_rate=0.1, random_state=42)
        m_D.fit(X_train, D_train)
        D_res[test_idx] = D_test - m_D.predict(X_test)
    
    print(f"    Fold {n_folds}/{n_folds} - Complete!   ")
    
    # Final stage: OLS on residuals
    theta = np.sum(D_res * Y_res) / np.sum(D_res * D_res)
    
    # Robust standard error
    residuals = Y_res - theta * D_res
    variance = np.mean((D_res * residuals)**2) / (np.mean(D_res**2)**2)
    se = np.sqrt(variance / len(D_res))
    
    ci_lower = theta - 1.96 * se
    ci_upper = theta + 1.96 * se
    t_stat = theta / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    
    print("  ✅ Double ML complete!")
    
    return {
        'ate': theta, 'se': se, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
        't_stat': t_stat, 'p_value': p_value, 'n_obs': len(data),
        'Y_residuals': Y_res, 'D_residuals': D_res
    }


def propensity_score_ipw(df, treatment='high_uncertainty', controls=None):
    """Inverse Probability Weighting using propensity scores."""
    print("\n📊 Running Propensity Score IPW...")
    
    if controls is None:
        controls = ['lag1_cash_flow_ratio', 'lag1_leverage', 'lag1_size', 
                   'lag1_profitability', 'News_Based_Policy_Uncert_Index']
    
    data = df[['investment_ratio', treatment] + controls].dropna().copy()
    
    Y = data['investment_ratio'].values
    D = data[treatment].values
    X = data[controls].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Estimate propensity scores
    ps_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                         learning_rate=0.1, random_state=42)
    ps_model.fit(X_scaled, D)
    propensity_scores = np.clip(ps_model.predict(X_scaled), 0.01, 0.99)
    
    # IPW weights
    weights = D / propensity_scores + (1 - D) / (1 - propensity_scores)
    
    # Weighted treatment effects
    y1_ipw = np.sum(weights * D * Y) / np.sum(weights * D)
    y0_ipw = np.sum(weights * (1 - D) * Y) / np.sum(weights * (1 - D))
    ate_ipw = y1_ipw - y0_ipw
    
    # Standard error
    ipw_residuals = weights * (D * (Y - y1_ipw) + (1 - D) * (Y - y0_ipw))
    se_ipw = np.sqrt(np.var(ipw_residuals) / len(Y))
    
    print("  ✅ IPW complete!")
    
    return {
        'ate_ipw': ate_ipw, 'se_ipw': se_ipw,
        'ci_lower': ate_ipw - 1.96 * se_ipw,
        'ci_upper': ate_ipw + 1.96 * se_ipw,
        'propensity_scores': propensity_scores
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_causal_results(dml_results, ps_results, panel_results, final_df):
    """Create comprehensive visualization of causal results."""
    print("\n📊 Creating causal inference visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Method comparison
    methods = ['Pooled OLS', 'Firm FE', 'Time FE', 'Two-way FE', 'Double ML', 'IPW']
    coefs = [
        panel_results['pooled'].params['high_uncertainty'],
        panel_results['firm_fe'].params['unc_fe'],
        panel_results['time_fe'].params['high_uncertainty'],
        panel_results['twoway_fe'].params['high_uncertainty_tw'],
        dml_results['ate'],
        ps_results['ate_ipw']
    ]
    ses = [
        panel_results['pooled'].bse['high_uncertainty'],
        panel_results['firm_fe'].bse['unc_fe'],
        panel_results['time_fe'].bse['high_uncertainty'],
        panel_results['twoway_fe'].bse['high_uncertainty_tw'],
        dml_results['se'],
        ps_results['se_ipw']
    ]
    
    colors = ['lightblue']*4 + ['coral', 'salmon']
    y_pos = np.arange(len(methods))
    
    axes[0,0].barh(y_pos, coefs, xerr=[1.96*s for s in ses], 
                   color=colors, alpha=0.7, capsize=5, edgecolor='black')
    axes[0,0].axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    axes[0,0].set_yticks(y_pos)
    axes[0,0].set_yticklabels(methods)
    axes[0,0].set_xlabel('Treatment Effect on Investment Ratio', fontsize=11)
    axes[0,0].set_title('Causal Estimates: Traditional vs ML Methods', fontweight='bold', fontsize=12)
    axes[0,0].grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Double ML residuals
    axes[0,1].scatter(dml_results['D_residuals'], dml_results['Y_residuals'], 
                     alpha=0.3, s=15, color='steelblue', edgecolors='none')
    slope = dml_results['ate']
    x_line = np.array([dml_results['D_residuals'].min(), dml_results['D_residuals'].max()])
    axes[0,1].plot(x_line, slope * x_line, 'r-', linewidth=2.5, 
                  label=f'ATE = {slope:.5f}')
    axes[0,1].set_xlabel('Treatment Residuals (D - E[D|X])', fontsize=11)
    axes[0,1].set_ylabel('Outcome Residuals (Y - E[Y|X])', fontsize=11)
    axes[0,1].set_title('Double ML: Partialled-Out Regression', fontweight='bold', fontsize=12)
    axes[0,1].legend(fontsize=10)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Propensity scores
    data_clean = final_df[['investment_ratio', 'high_uncertainty', 'lag1_cash_flow_ratio', 
                           'lag1_leverage', 'lag1_size', 'lag1_profitability', 
                           'News_Based_Policy_Uncert_Index']].dropna()
    
    treated_ps = ps_results['propensity_scores'][data_clean['high_uncertainty'] == 1]
    control_ps = ps_results['propensity_scores'][data_clean['high_uncertainty'] == 0]
    
    axes[1,0].hist([control_ps, treated_ps], bins=30, alpha=0.7,
                   label=['Control', 'Treated'], color=['lightblue', 'lightcoral'], 
                   edgecolor='black')
    axes[1,0].axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    axes[1,0].set_xlabel('Propensity Score', fontsize=11)
    axes[1,0].set_ylabel('Frequency', fontsize=11)
    axes[1,0].set_title('Propensity Score Distribution', fontweight='bold', fontsize=12)
    axes[1,0].legend(fontsize=10)
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    axes[1,1].axis('off')
    summary_data = [
        ['Method', 'ATE', 'Std Err', 'P-value', '95% CI'],
        ['─'*12, '─'*9, '─'*8, '─'*8, '─'*22],
        ['Double ML', f'{dml_results["ate"]:.5f}', f'{dml_results["se"]:.5f}', 
         f'{dml_results["p_value"]:.4f}', 
         f'[{dml_results["ci_lower"]:.5f}, {dml_results["ci_upper"]:.5f}]'],
        ['IPW', f'{ps_results["ate_ipw"]:.5f}', f'{ps_results["se_ipw"]:.5f}', 
         '─', 
         f'[{ps_results["ci_lower"]:.5f}, {ps_results["ci_upper"]:.5f}]'],
        ['Two-way FE', f'{coefs[3]:.5f}', f'{ses[3]:.5f}', 
         f'{panel_results["twoway_fe"].pvalues["high_uncertainty_tw"]:.4f}', '─']
    ]
    
    table = axes[1,1].table(cellText=summary_data, cellLoc='left',
                           loc='center', bbox=[0, 0.25, 1, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#2E7D32')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1,1].set_title('Causal Inference Results Summary', 
                       fontweight='bold', pad=20, fontsize=13)
    
    plt.tight_layout()
    plt.savefig('plots/causal_inference_complete.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ✅ Visualizations saved!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("POLITICAL UNCERTAINTY & CORPORATE INVESTMENT")
    print("Panel Econometrics + Double Machine Learning")
    print("="*70)
    
    # Paths
    epu_path = 'data/US_Policy_Uncertainty_Data.xlsx'
    balance_path = 'data/us-balance-quarterly.csv'
    cashflow_path = 'data/us-cashflow-quarterly.csv'
    income_path = 'data/us-income-quarterly.csv'
    
    # Load and prepare data
    epu_quarterly = load_and_process_epu_data(epu_path)
    financial_data = load_and_merge_financial_data(balance_path, cashflow_path, income_path)
    financial_data = create_analysis_variables(financial_data)
    final_df = merge_epu_and_financial(epu_quarterly, financial_data)
    
    # Exploratory analysis
    min_year = financial_data['report_date'].min().year
    epu_filtered = epu_quarterly[epu_quarterly['date'].dt.year >= min_year]
    exploratory_analysis(final_df, epu_filtered)
    
    # Panel econometrics
    panel_results = run_panel_econometrics(final_df)
    
    # Print panel results
    print("\n" + "="*70)
    print("PANEL REGRESSION RESULTS")
    print("="*70)
    
    models = [
        ('Pooled OLS', panel_results['pooled'], 'high_uncertainty'),
        ('Firm FE', panel_results['firm_fe'], 'unc_fe'),
        ('Time FE', panel_results['time_fe'], 'high_uncertainty'),
        ('Two-way FE + Controls', panel_results['twoway_fe'], 'high_uncertainty_tw')
    ]
    
    for name, model, var in models:
        print(f"\n{name}:")
        print(f"  Coefficient: {model.params[var]:.6f}")
        print(f"  Std Error: {model.bse[var]:.6f}")
        print(f"  P-value: {model.pvalues[var]:.4f}")
        print(f"  R-squared: {model.rsquared:.4f}")
        print(f"  Observations: {int(model.nobs)}")
    
    # Causal machine learning
    print("\n" + "="*70)
    print("CAUSAL MACHINE LEARNING")
    print("="*70)
    
    dml_results = double_ml_estimation(final_df)
    ps_results = propensity_score_ipw(final_df)
    
    # Print causal ML results
    print("\n" + "="*70)
    print("CAUSAL INFERENCE RESULTS")
    print("="*70)
    
    print("\n🤖 Double Machine Learning (DML):")
    print(f"  Average Treatment Effect: {dml_results['ate']:.6f}")
    print(f"  Standard Error: {dml_results['se']:.6f}")
    print(f"  95% Confidence Interval: [{dml_results['ci_lower']:.6f}, {dml_results['ci_upper']:.6f}]")
    print(f"  T-statistic: {dml_results['t_stat']:.4f}")
    print(f"  P-value: {dml_results['p_value']:.4f}")
    print(f"  {'✅ Significant at 5% level' if dml_results['p_value'] < 0.05 else '❌ Not significant'}")
    
    print("\n📊 Inverse Probability Weighting (IPW):")
    print(f"  Average Treatment Effect: {ps_results['ate_ipw']:.6f}")
    print(f"  Standard Error: {ps_results['se_ipw']:.6f}")
    print(f"  95% Confidence Interval: [{ps_results['ci_lower']:.6f}, {ps_results['ci_upper']:.6f}]")
    
    print("\n📈 Interpretation:")
    if dml_results['ate'] < 0:
        print(f"  High political uncertainty REDUCES corporate investment by")
        print(f"  {abs(dml_results['ate'])*100:.3f} percentage points (DML estimate)")
    else:
        print(f"  High political uncertainty INCREASES corporate investment by")
        print(f"  {dml_results['ate']*100:.3f} percentage points (DML estimate)")
    
    # Visualize
    plot_causal_results(dml_results, ps_results, panel_results, final_df)
    
    # Save results
    final_df.to_csv('results/causal_analysis_data.csv', index=False)
    
    # Save summary
    summary_df = pd.DataFrame({
        'Method': ['Pooled OLS', 'Firm FE', 'Time FE', 'Two-way FE', 'Double ML', 'IPW'],
        'Coefficient': [
            panel_results['pooled'].params['high_uncertainty'],
            panel_results['firm_fe'].params['unc_fe'],
            panel_results['time_fe'].params['high_uncertainty'],
            panel_results['twoway_fe'].params['high_uncertainty_tw'],
            dml_results['ate'],
            ps_results['ate_ipw']
        ],
        'Std_Error': [
            panel_results['pooled'].bse['high_uncertainty'],
            panel_results['firm_fe'].bse['unc_fe'],
            panel_results['time_fe'].bse['high_uncertainty'],
            panel_results['twoway_fe'].bse['high_uncertainty_tw'],
            dml_results['se'],
            ps_results['se_ipw']
        ]
    })
    summary_df.to_csv('results/estimation_results.csv', index=False)
    
    print("\n💾 Results saved to 'results/' directory")
    print("\n🎉 Analysis Complete!")
    print("="*70)
    
    return final_df, panel_results, dml_results, ps_results


if __name__ == "__main__":
    final_df, panel_results, dml_results, ps_results = main()