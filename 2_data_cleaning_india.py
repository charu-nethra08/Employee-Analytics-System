import pandas as pd
import numpy as np
import json

def clean_data(filepath):
    report = {}
    df = pd.read_csv(filepath)
    report['original_shape'] = list(df.shape)
    report['original_missing'] = int(df.isnull().sum().sum())
    report['original_duplicates'] = int(df.duplicated().sum())

    df = df.drop_duplicates()
    report['after_dedup_rows'] = int(len(df))

    df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')
    df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    def remove_outliers(df, col, low_bound=None, high_bound=None):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = low_bound if low_bound is not None else Q1 - 3 * IQR
        upper = high_bound if high_bound is not None else Q3 + 3 * IQR
        original = len(df)
        df = df[(df[col].isna()) | ((df[col] >= lower) & (df[col] <= upper))]
        return df, original - len(df)

    # INR bounds: 3L to 30L
    df, salary_removed = remove_outliers(df, 'salary', low_bound=300000, high_bound=3000000)
    df, age_removed = remove_outliers(df, 'age', low_bound=18, high_bound=65)
    df, exp_removed = remove_outliers(df, 'years_experience', low_bound=0, high_bound=40)

    report['outliers_removed'] = {'salary': int(salary_removed), 'age': int(age_removed), 'years_experience': int(exp_removed)}

    for col in ['salary', 'performance_score', 'satisfaction_score']:
        median_vals = df.groupby('department')[col].transform('median')
        df[col] = df[col].fillna(median_vals)

    df['seniority'] = pd.cut(df['years_experience'], bins=[-1,2,7,15,100], labels=['Junior','Mid','Senior','Principal'])
    df['salary_band'] = pd.cut(df['salary'], bins=[0,500000,900000,1500000,3000000], labels=['Entry','Mid','Upper-Mid','Senior'])
    df['hire_year'] = df['hire_date'].dt.year
    df['performance_tier'] = pd.cut(df['performance_score'], bins=[0,2,3.5,5], labels=['Low','Medium','High'])

    report['final_shape'] = list(df.shape)
    report['final_missing'] = int(df.isnull().sum().sum())

    stats = {}
    stats['total_employees'] = int(len(df))
    stats['avg_salary'] = round(float(df['salary'].mean()), 0)
    stats['avg_performance'] = round(float(df['performance_score'].mean()), 2)
    stats['avg_satisfaction'] = round(float(df['satisfaction_score'].mean()), 2)

    dept_stats = df.groupby('department').agg(
        count=('employee_id','count'), avg_salary=('salary','mean'),
        avg_perf=('performance_score','mean'), avg_sat=('satisfaction_score','mean'),
        avg_projects=('projects_completed','mean')
    ).round(0).reset_index()
    stats['by_department'] = dept_stats.to_dict(orient='records')

    stats['by_gender'] = df['gender'].value_counts().reset_index().rename(columns={'gender':'gender','count':'count'}).to_dict(orient='records')

    seniority_stats = df.groupby('seniority', observed=True).agg(
        count=('employee_id','count'), avg_salary=('salary','mean'), avg_perf=('performance_score','mean')
    ).round(0).reset_index()
    stats['by_seniority'] = seniority_stats.to_dict(orient='records')

    stats['by_performance_tier'] = df['performance_tier'].value_counts().reset_index().rename(columns={'performance_tier':'tier','count':'count'}).to_dict(orient='records')

    remote_dist = df['remote_days'].value_counts().sort_index().reset_index()
    remote_dist.columns = ['days','count']
    stats['remote_distribution'] = remote_dist.to_dict(orient='records')

    salary_hist, salary_edges = np.histogram(df['salary'].dropna(), bins=10)
    stats['salary_histogram'] = {'counts': salary_hist.tolist(), 'edges': [round(e) for e in salary_edges.tolist()]}

    hire_trend = df.groupby('hire_year')['employee_id'].count().reset_index()
    hire_trend.columns = ['year','count']
    stats['hire_trend'] = hire_trend.dropna().to_dict(orient='records')

    corr_data = df[['years_experience','salary','performance_score','satisfaction_score','training_hours']].dropna()
    stats['correlation'] = {
        'exp_salary': round(float(corr_data['years_experience'].corr(corr_data['salary'])), 3),
        'exp_perf': round(float(corr_data['years_experience'].corr(corr_data['performance_score'])), 3),
        'sat_perf': round(float(corr_data['satisfaction_score'].corr(corr_data['performance_score'])), 3),
        'training_perf': round(float(corr_data['training_hours'].corr(corr_data['performance_score'])), 3),
    }

    top_earners = df.nlargest(5,'salary')[['name','department','city','salary','years_experience']].copy()
    top_earners['salary'] = top_earners['salary'].round(0)
    stats['top_earners'] = top_earners.to_dict(orient='records')

    # City breakdown
    city_counts = df['city'].value_counts().head(8).reset_index()
    city_counts.columns = ['city','count']
    stats['by_city'] = city_counts.to_dict(orient='records')

    df.to_csv('/home/claude/cleaned_india.csv', index=False)
    with open('/home/claude/stats_india.json','w') as f:
        json.dump(stats, f, indent=2, default=str)
    with open('/home/claude/report_india.json','w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nEmployees: {stats['total_employees']}, Avg Salary: ₹{stats['avg_salary']:,.0f}")
    return df, stats, report

df, stats, report = clean_data('/home/claude/raw_employee_data_india.csv')
