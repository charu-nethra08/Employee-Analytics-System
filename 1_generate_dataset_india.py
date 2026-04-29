import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker('en_IN')
Faker.seed(42)
np.random.seed(42)
random.seed(42)

n = 500
departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations']
genders = ['Male', 'Female', 'Non-binary']

# Indian cities
cities = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata',
          'Ahmedabad', 'Jaipur', 'Surat', 'Lucknow', 'Chandigarh', 'Noida', 'Gurgaon', 'Kochi']

# Salary in INR (realistic Indian IT/corporate range: 3L - 25L per annum)
salaries_inr = np.random.normal(900000, 400000, n)  # mean ~9 LPA

data = {
    'employee_id': [f'EMP{str(i).zfill(4)}' for i in range(1, n+1)],
    'name': [fake.name() for _ in range(n)],
    'age': np.random.randint(22, 60, n).tolist(),
    'gender': [random.choice(genders) for _ in range(n)],
    'department': [random.choice(departments) for _ in range(n)],
    'salary': salaries_inr.tolist(),
    'years_experience': np.random.randint(0, 30, n).tolist(),
    'performance_score': np.random.uniform(1, 5, n).tolist(),
    'satisfaction_score': np.random.uniform(1, 10, n).tolist(),
    'remote_days': np.random.randint(0, 6, n).tolist(),
    'projects_completed': np.random.poisson(8, n).tolist(),
    'training_hours': np.random.exponential(40, n).tolist(),
    'city': [random.choice(cities) for _ in range(n)],
    'hire_date': [fake.date_between(start_date='-15y', end_date='today').strftime('%Y-%m-%d') for _ in range(n)],
}

df = pd.DataFrame(data)

# Inject issues
missing_indices = np.random.choice(n, 60, replace=False)
df.loc[missing_indices[:20], 'salary'] = np.nan
df.loc[missing_indices[20:40], 'performance_score'] = np.nan
df.loc[missing_indices[40:], 'satisfaction_score'] = np.nan

outlier_indices = np.random.choice(n, 15, replace=False)
df.loc[outlier_indices[:5], 'salary'] = np.random.choice([5000000, 6000000, -50000, 0, 10000], 5)
df.loc[outlier_indices[5:10], 'age'] = np.random.choice([150, 200, -5, 1, 0], 5)
df.loc[outlier_indices[10:], 'years_experience'] = np.random.choice([100, 80, -10], 5)

dup_indices = np.random.choice(n, 20, replace=False)
df = pd.concat([df, df.iloc[dup_indices]], ignore_index=True)

df.to_csv('/home/claude/raw_employee_data_india.csv', index=False)
print(f"Dataset: {len(df)} rows")
print("Sample names:", df['name'].head(5).tolist())
print("Sample cities:", df['city'].head(5).tolist())
print(f"Salary range: ₹{df['salary'].min():.0f} – ₹{df['salary'].max():.0f}")
