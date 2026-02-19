"""
edo.py
Exploratory Data Analysis Module for LA Crime Data
EAS 587 - Phase 1 Project
Following John Tukey's EDA Principles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_cleaned_data(filepath):
    """Load cleaned crime data."""
    print("Loading cleaned data...")
    df = pd.read_csv(filepath, parse_dates=['Date Rptd', 'DATE OCC'])
    print(f"Loaded {len(df):,} records")
    return df

# ============================================
# EDA OPERATION 1: Summary Statistics
# ============================================
def eda_summary_statistics(df):
    """
    EDA 1: Generate comprehensive summary statistics
    Following Tukey's principle of understanding data through summaries
    """
    print("\n" + "="*60)
    print("EDA 1: SUMMARY STATISTICS")
    print("="*60)

    # Numeric columns summary
    numeric_cols = ['Vict Age', 'Hour', 'Reporting Delay (Days)']
    print("\nNumeric Columns Summary:")
    print(df[numeric_cols].describe())

    # Categorical summaries
    print("\nCrime Categories:")
    print(df['Crime Category'].value_counts())

    print("\nPremise Categories:")
    print(df['Premise Category'].value_counts())

    return df[numeric_cols].describe()

# ============================================
# EDA OPERATION 2: Temporal Analysis
# ============================================
def eda_temporal_patterns(df):
    """
    EDA 2: Analyze temporal patterns in crime
    - Crime by hour of day
    - Crime by day of week
    - Crime by month
    """
    print("\n" + "="*60)
    print("EDA 2: TEMPORAL PATTERNS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Crime by Hour
    hourly_crimes = df['Hour'].value_counts().sort_index()
    axes[0, 0].bar(hourly_crimes.index, hourly_crimes.values, color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Crimes by Hour of Day', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Number of Crimes')
    axes[0, 0].set_xticks(range(0, 24, 2))

    # Crime by Day of Week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_crimes = df['DayOfWeek'].value_counts().reindex(day_order)
    axes[0, 1].bar(range(7), daily_crimes.values, color='coral', edgecolor='black')
    axes[0, 1].set_title('Crimes by Day of Week', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Number of Crimes')
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    # Crime by Month
    monthly_crimes = df['Month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[1, 0].plot(monthly_crimes.index, monthly_crimes.values, marker='o', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_title('Crimes by Month (2024)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Number of Crimes')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_xticklabels(month_names)
    axes[1, 0].grid(True, alpha=0.3)

    # Top Crime Types
    top_crimes = df['Crm Cd Desc'].value_counts().head(10)
    axes[1, 1].barh(range(len(top_crimes)), top_crimes.values, color='purple', edgecolor='black')
    axes[1, 1].set_title('Top 10 Crime Types', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Crimes')
    axes[1, 1].set_yticks(range(len(top_crimes)))
    axes[1, 1].set_yticklabels([label[:30] + '...' if len(label) > 30 else label for label in top_crimes.index], fontsize=8)
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('figures/temporal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  - Saved temporal patterns visualization")
    print(f"  - Peak crime hour: {hourly_crimes.idxmax()}:00 ({hourly_crimes.max():,} crimes)")
    print(f"  - Highest crime day: {daily_crimes.idxmax()} ({daily_crimes.max():,} crimes)")

    return hourly_crimes, daily_crimes, monthly_crimes

# ============================================
# EDA OPERATION 3: Geographic Analysis
# ============================================
def eda_geographic_distribution(df):
    """
    EDA 3: Analyze geographic distribution of crimes
    - Crime by area
    - Crime density visualization
    """
    print("\n" + "="*60)
    print("EDA 3: GEOGRAPHIC DISTRIBUTION")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Crime by Area
    area_crimes = df['AREA NAME'].value_counts()
    axes[0].barh(range(len(area_crimes)), area_crimes.values, color='teal', edgecolor='black')
    axes[0].set_title('Crimes by LAPD Area', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Crimes')
    axes[0].set_yticks(range(len(area_crimes)))
    axes[0].set_yticklabels(area_crimes.index, fontsize=9)
    axes[0].invert_yaxis()

    # Crime coordinates scatter (sample for performance)
    sample_df = df[df['Valid Coordinates']].sample(min(5000, len(df)), random_state=42)
    scatter = axes[1].scatter(sample_df['LON'], sample_df['LAT'], 
                              c=sample_df['Crm Cd'], cmap='tab10', 
                              alpha=0.5, s=10)
    axes[1].set_title('Crime Locations (Sample)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')

    plt.tight_layout()
    plt.savefig('figures/geographic_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  - Saved geographic distribution visualization")
    print(f"  - Highest crime area: {area_crimes.idxmax()} ({area_crimes.max():,} crimes)")

    return area_crimes

# ============================================
# EDA OPERATION 4: Victim Demographics
# ============================================
def eda_victim_demographics(df):
    """
    EDA 4: Analyze victim demographics
    - Age distribution
    - Sex distribution
    - Ethnicity distribution
    """
    print("\n" + "="*60)
    print("EDA 4: VICTIM DEMOGRAPHICS")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Age distribution
    valid_ages = df[df['Vict Age'].notna() & (df['Vict Age'] > 0) & (df['Vict Age'] < 100)]
    axes[0, 0].hist(valid_ages['Vict Age'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Victim Age Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(valid_ages['Vict Age'].median(), color='red', linestyle='--', 
                       label=f'Median: {valid_ages["Vict Age"].median():.1f}')
    axes[0, 0].legend()

    # Age group distribution
    age_group_counts = df['Age Group'].value_counts().sort_index()
    axes[0, 1].bar(range(len(age_group_counts)), age_group_counts.values, 
                   color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Crimes by Age Group', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Age Group')
    axes[0, 1].set_ylabel('Number of Crimes')
    axes[0, 1].set_xticks(range(len(age_group_counts)))
    axes[0, 1].set_xticklabels(age_group_counts.index, rotation=45)

    # Sex distribution
    sex_counts = df['Vict Sex Clean'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    axes[1, 0].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
    axes[1, 0].set_title('Victim Sex Distribution', fontsize=12, fontweight='bold')

    # Descent distribution (top 8)
    descent_counts = df['Vict Descent Clean'].value_counts().head(8)
    axes[1, 1].bar(range(len(descent_counts)), descent_counts.values, 
                   color='orange', edgecolor='black')
    axes[1, 1].set_title('Victim Descent (Top 8)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Descent')
    axes[1, 1].set_ylabel('Number of Crimes')
    axes[1, 1].set_xticks(range(len(descent_counts)))
    axes[1, 1].set_xticklabels(descent_counts.index, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('figures/victim_demographics.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  - Saved victim demographics visualization")
    print(f"  - Median victim age: {valid_ages['Vict Age'].median():.1f}")
    print(f"  - Most common victim sex: {sex_counts.idxmax()} ({sex_counts.max():,})")

    return valid_ages, sex_counts, descent_counts

# ============================================
# EDA OPERATION 5: Crime Type Analysis
# ============================================
def eda_crime_type_analysis(df):
    """
    EDA 5: Deep dive into crime types and categories
    """
    print("\n" + "="*60)
    print("EDA 5: CRIME TYPE ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Crime categories
    crime_cat = df['Crime Category'].value_counts()
    axes[0].barh(range(len(crime_cat)), crime_cat.values, color='indianred', edgecolor='black')
    axes[0].set_title('Crimes by Category', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Crimes')
    axes[0].set_yticks(range(len(crime_cat)))
    axes[0].set_yticklabels(crime_cat.index, fontsize=9)
    axes[0].invert_yaxis()

    # Premise categories
    premise_cat = df['Premise Category'].value_counts()
    axes[1].barh(range(len(premise_cat)), premise_cat.values, color='mediumseagreen', edgecolor='black')
    axes[1].set_title('Crimes by Premise Type', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Crimes')
    axes[1].set_yticks(range(len(premise_cat)))
    axes[1].set_yticklabels(premise_cat.index, fontsize=9)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('figures/crime_type_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  - Saved crime type analysis visualization")

    return crime_cat, premise_cat

# ============================================
# EDA OPERATION 6: Reporting Patterns
# ============================================
def eda_reporting_patterns(df):
    """
    EDA 6: Analyze crime reporting patterns
    - Reporting delay distribution
    - Status of investigations
    """
    print("\n" + "="*60)
    print("EDA 6: REPORTING PATTERNS")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Reporting delay
    delays = df[df['Reporting Delay (Days)'] <= 30]['Reporting Delay (Days)']
    axes[0].hist(delays, bins=30, color='gold', edgecolor='black')
    axes[0].set_title('Reporting Delay Distribution (≤30 days)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Days Between Crime and Report')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(delays.median(), color='red', linestyle='--',
                    label=f'Median: {delays.median():.1f} days')
    axes[0].legend()

    # Case status
    status_counts = df['Status Desc'].value_counts()
    axes[1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                startangle=90)
    axes[1].set_title('Case Status Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/reporting_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  - Saved reporting patterns visualization")
    print(f"  - Median reporting delay: {delays.median():.1f} days")

    return delays, status_counts

# ============================================
# EDA OPERATION 7: Cross-tabulation Analysis
# ============================================
def eda_cross_tabulation(df):
    """
    EDA 7: Cross-tabulation analysis
    - Crime category by victim sex
    - Crime by area and time
    """
    print("\n" + "="*60)
    print("EDA 7: CROSS-TABULATION ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Crime category by sex
    cross_tab = pd.crosstab(df['Crime Category'], df['Vict Sex Clean'], normalize='columns') * 100
    cross_tab.plot(kind='bar', ax=axes[0], stacked=True)
    axes[0].set_title('Crime Category by Victim Sex (%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Crime Category')
    axes[0].set_ylabel('Percentage')
    axes[0].legend(title='Sex', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].tick_params(axis='x', rotation=45)

    # Top areas by crime category
    top_areas = df['AREA NAME'].value_counts().head(5).index
    area_crime_cross = pd.crosstab(df[df['AREA NAME'].isin(top_areas)]['AREA NAME'], 
                                    df[df['AREA NAME'].isin(top_areas)]['Crime Category'])
    area_crime_cross.plot(kind='bar', ax=axes[1], stacked=True)
    axes[1].set_title('Crime Categories by Top 5 Areas', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Area')
    axes[1].set_ylabel('Number of Crimes')
    axes[1].legend(title='Crime Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('figures/cross_tabulation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  - Saved cross-tabulation visualization")

    return cross_tab, area_crime_cross

# ============================================
# EDA OPERATION 8: Correlation Analysis
# ============================================
def eda_correlation_analysis(df):
    """
    EDA 8: Correlation analysis of numeric variables
    """
    print("\n" + "="*60)
    print("EDA 8: CORRELATION ANALYSIS")
    print("="*60)

    # Select numeric columns
    numeric_df = df[['Vict Age', 'Hour', 'Month', 'Reporting Delay (Days)', 'LAT', 'LON']].copy()

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax, fmt='.2f')
    ax.set_title('Correlation Matrix of Numeric Variables', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  - Saved correlation matrix visualization")
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(2))

    return corr_matrix

# ============================================
# EDA OPERATION 9: Outlier Detection
# ============================================
def eda_outlier_detection(df):
    """
    EDA 9: Detect and visualize outliers using box plots
    """
    print("\n" + "="*60)
    print("EDA 9: OUTLIER DETECTION")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Age outliers
    valid_ages = df[(df['Vict Age'].notna()) & (df['Vict Age'] > 0) & (df['Vict Age'] < 100)]['Vict Age']
    axes[0].boxplot(valid_ages, vert=True)
    axes[0].set_title('Age Distribution (Box Plot)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Age')

    # Reporting delay outliers
    delay_data = df[df['Reporting Delay (Days)'] <= 100]['Reporting Delay (Days)']
    axes[1].boxplot(delay_data, vert=True)
    axes[1].set_title('Reporting Delay (Box Plot)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Days')

    # Hour distribution
    axes[2].boxplot(df['Hour'], vert=True)
    axes[2].set_title('Hour Distribution (Box Plot)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Hour')

    plt.tight_layout()
    plt.savefig('figures/outlier_detection.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  - Saved outlier detection visualization")

    # Calculate IQR for age
    Q1 = valid_ages.quantile(0.25)
    Q3 = valid_ages.quantile(0.75)
    IQR = Q3 - Q1
    outliers = valid_ages[(valid_ages < (Q1 - 1.5 * IQR)) | (valid_ages > (Q3 + 1.5 * IQR))]
    print(f"  - Age outliers (IQR method): {len(outliers)} records")

    return outliers

# ============================================
# EDA OPERATION 10: Weapon Usage Analysis
# ============================================
def eda_weapon_analysis(df):
    """
    EDA 10: Analyze weapon usage in crimes
    """
    print("\n" + "="*60)
    print("EDA 10: WEAPON USAGE ANALYSIS")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Weapon usage (has weapon vs no weapon)
    has_weapon = df['Weapon Used Cd'].notna().sum()
    no_weapon = df['Weapon Used Cd'].isna().sum()

    weapon_summary = pd.Series({'No Weapon': no_weapon, 'Weapon Used': has_weapon})
    axes[0].pie(weapon_summary.values, labels=weapon_summary.index, autopct='%1.1f%%',
                colors=['lightblue', 'salmon'], startangle=90)
    axes[0].set_title('Weapon Usage in Crimes', fontsize=12, fontweight='bold')

    # Top weapon types
    top_weapons = df['Weapon Desc'].value_counts().head(8)
    axes[1].barh(range(len(top_weapons)), top_weapons.values, color='crimson', edgecolor='black')
    axes[1].set_title('Top Weapon Types', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Crimes')
    axes[1].set_yticks(range(len(top_weapons)))
    axes[1].set_yticklabels(top_weapons.index, fontsize=9)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('figures/weapon_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  - Saved weapon analysis visualization")
    print(f"  - Crimes with weapons: {has_weapon:,} ({has_weapon/len(df)*100:.1f}%)")

    return weapon_summary, top_weapons

def main():
    """Main EDA pipeline."""
    # Load cleaned data
    df = load_cleaned_data('data/processed/crime_data_cleaned.csv')

    # Run all EDA operations
    eda_summary_statistics(df)
    eda_temporal_patterns(df)
    eda_geographic_distribution(df)
    eda_victim_demographics(df)
    eda_crime_type_analysis(df)
    eda_reporting_patterns(df)
    eda_cross_tabulation(df)
    eda_correlation_analysis(df)
    eda_outlier_detection(df)
    eda_weapon_analysis(df)

    print("\n" + "="*60)
    print("EDA COMPLETE - All visualizations saved to figures/")
    print("="*60)

if __name__ == "__main__":
    main()
name__ == "__main__":
    main()
