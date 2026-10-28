"""Transparent batch data quality, drift and performance calculations."""
from collections import Counter
from io import BytesIO
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

MAX_BYTES = 20 * 1024 * 1024


def read_csv(data: bytes) -> pd.DataFrame:
    """Read CSV with size, header, and basic structural validation."""
    import csv
    import io
    if not data or len(data) > MAX_BYTES:
        raise ValueError("CSV is empty or exceeds the 20 MB limit.")
    try:
        header = next(csv.reader(io.StringIO(data.decode('utf-8-sig'))))
        if not header or any(not x.strip() for x in header) or len(set(header)) != len(header):
            raise ValueError("Columns must have unique, non-empty names.")
        df = pd.read_csv(BytesIO(data))
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration) as exc:
        raise ValueError("Invalid UTF-8 CSV file.") from exc
    if df.empty:
        raise ValueError("Dataset has no rows.")
    if len(df.columns) > 500 or len(df) > 200000:
        raise ValueError("Dataset exceeds the 500 column or 200,000 row limit.")
    return df


def profile(df: pd.DataFrame) -> dict:
    """Return quality summary without retaining raw values."""
    rows, cols = df.shape
    columns = []
    for name in df.columns:
        s = df[name]
        numeric = pd.api.types.is_numeric_dtype(s)
        inf = int(np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).sum()) if numeric else 0
        columns.append({'column': name, 'dtype': str(s.dtype), 'missing': int(s.isna().sum()),
                        'missing_pct': float(s.isna().mean() * 100), 'unique': int(s.nunique()),
                        'infinite': inf, 'constant': bool(s.nunique() <= 1)})
    return {'rows': rows, 'columns': cols, 'duplicate_rows': int(df.duplicated().sum()),
            'missing_pct': float(df.isna().to_numpy().mean() * 100), 'features': columns}


def analyze(reference: pd.DataFrame, current: pd.DataFrame, features: list[str], alpha: float = 0.05) -> dict:
    """Compare distributions using KS or chi-square, explicitly recording skipped features."""
    if not 0 < alpha < 1 or not features:
        raise ValueError('Select at least one feature and a threshold between 0 and 1.')
    missing = sorted(set(reference.columns) - set(current.columns))
    added = sorted(set(current.columns) - set(reference.columns))
    results = []
    for name in features:
        item = {'feature': name, 'status': 'unavailable', 'method': None, 'p_value': None, 'drift': None, 'reason': None}
        if name not in reference or name not in current:
            item['reason'] = 'Feature absent from one dataset.'
            results.append(item); continue
        a, b = reference[name].dropna(), current[name].dropna()
        an, bn = pd.api.types.is_numeric_dtype(a), pd.api.types.is_numeric_dtype(b)
        if an != bn:
            item['reason'] = 'Incompatible numeric and categorical types.'
        elif an:
            a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
            if len(a) < 2 or len(b) < 2:
                item['reason'] = 'Fewer than two finite values in a dataset.'
            else:
                result = ks_2samp(a, b)
                item.update(status='ok', method='Two-sample KS', p_value=float(result.pvalue), statistic=float(result.statistic))
        elif len(a) < 2 or len(b) < 2:
            item['reason'] = 'Fewer than two non-null values in a dataset.'
        else:
            ca, cb = Counter(a.astype(str)), Counter(b.astype(str))
            cats = sorted(set(ca) | set(cb))
            if len(cats) > 100:
                item['reason'] = 'More than 100 categories; exclude identifiers or group categories.'
            elif len(cats) < 2:
                item.update(status='ok', method='Chi-square', p_value=1.0, statistic=0.0, reason='Both datasets have the same constant category.')
            else:
                table = np.array([[ca[x] for x in cats], [cb[x] for x in cats]])
                stat, p, _, expected = chi2_contingency(table, correction=False)
                item.update(status='ok', method='Chi-square', p_value=float(p), statistic=float(stat),
                            reason='Expected counts below 5; interpret cautiously.' if (expected < 5).any() else None)
        if item['status'] == 'ok':
            item['drift'] = item['p_value'] < alpha
        results.append(item)
    valid = [x for x in results if x['status'] == 'ok']
    return {'methodology': 'KS for numeric; chi-square for categorical; no multiple-testing correction', 'alpha': alpha,
            'missing_columns': missing, 'added_columns': added, 'features': results,
            'analyzed': len(valid), 'selected': len(features), 'drifted': sum(x['drift'] for x in valid),
            'drift_pct': 100 * sum(x['drift'] for x in valid) / len(valid) if valid else None,
            'coverage_complete': len(valid) == len(features), 'reference_quality': profile(reference), 'current_quality': profile(current)}


def evaluate(df: pd.DataFrame, target: str, prediction: str, task: str) -> dict:
    if target == prediction or target not in df or prediction not in df:
        raise ValueError('Choose distinct existing label and prediction columns.')
    pairs = df[[target, prediction]].dropna()
    if pairs.empty:
        raise ValueError('No rows contain both actual labels and predictions.')
    y, p = pairs[target], pairs[prediction]
    out = {'task': task, 'rows_used': len(pairs), 'rows_excluded': len(df) - len(pairs)}
    if task == 'classification':
        if pd.api.types.is_numeric_dtype(y) and pd.api.types.is_numeric_dtype(p):
            y, p = y.astype(float), p.astype(float)
            if not np.isfinite(y).all() or not np.isfinite(p).all():
                raise ValueError('Classification labels and predictions must be finite.')
        else:
            y, p = y.astype(str), p.astype(str)
        precision, recall, f1, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
        labels = sorted(set(y) | set(p))
        out.update(accuracy=float(accuracy_score(y,p)), precision_weighted=float(precision), recall_weighted=float(recall), f1_weighted=float(f1), labels=labels, confusion_matrix=confusion_matrix(y,p,labels=labels).tolist())
    elif task == 'regression':
        y, p = pd.to_numeric(y, errors='coerce'), pd.to_numeric(p, errors='coerce')
        if not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError('Regression labels and predictions must be finite numeric values.')
        mse = float(mean_squared_error(y,p))
        out.update(mae=float(mean_absolute_error(y,p)), mse=mse, rmse=math.sqrt(mse), r2=float(r2_score(y,p)) if len(y) >= 2 and y.nunique() > 1 else None)
    else:
        raise ValueError('Unknown task.')
    return out


def serialize(result: dict) -> str:
    return json.dumps(result, indent=2, allow_nan=False, default=str)
