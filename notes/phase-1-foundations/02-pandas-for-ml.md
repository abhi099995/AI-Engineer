# Pandas for ML

## Why Pandas for ML?
- Structured data manipulation (DataFrames)
- The bridge between raw data and model training
- Almost every ML workflow starts here

## Key Concepts

### DataFrame Basics
```python
import pandas as pd

df = pd.read_csv("data.csv")
df.head()           # first 5 rows
df.shape            # (rows, cols)
df.dtypes           # data types per column
df.describe()       # statistical summary
df.info()           # null counts and types
```

### Selecting Data
```python
df["column"]                    # single column (Series)
df[["col1", "col2"]]            # multiple columns
df.loc[0:5, "col"]              # label-based
df.iloc[0:5, 0:3]               # position-based
df[df["age"] > 30]              # filtering
df.query("age > 30 and salary > 50000")
```

### Handling Missing Values
```python
df.isnull().sum()               # count nulls per column
df.dropna()                     # drop rows with any null
df.fillna(0)                    # fill with a value
df["col"].fillna(df["col"].mean())  # fill with mean
```

### Feature Engineering Patterns
```python
# Create new columns
df["age_squared"] = df["age"] ** 2
df["is_senior"] = (df["age"] > 60).astype(int)

# Binning
df["age_bucket"] = pd.cut(df["age"], bins=[0, 25, 45, 65, 100],
                           labels=["young", "adult", "middle", "senior"])

# One-hot encoding
df = pd.get_dummies(df, columns=["category"], drop_first=True)

# Groupby aggregations
df.groupby("region")["revenue"].agg(["mean", "sum", "count"])
```

### Train/Test Splitting (with pandas)
```python
train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

# Or use sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("target", axis=1), df["target"], test_size=0.2, random_state=42
)
```

## Common Mistakes
- Never fill null values with mean on the entire dataset before splitting (data leakage)
- `loc` is label-based, `iloc` is position-based — easy to confuse

## Personal Notes
- TODO: Practice a full EDA (exploratory data analysis) pipeline on a real dataset
