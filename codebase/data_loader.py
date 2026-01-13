import pandas as pd
import numpy as np
import os
import requests
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataLoader:
    def __init__(self, dataset_name, random_state=42):
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.data = None
        self.target_col = None
        self.categorical_cols = []
        self.numerical_cols = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Ensure cache directory exists
        self.cache_dir = 'datasets'
        os.makedirs(self.cache_dir, exist_ok=True)

    def _download_with_progress(self, url, dest_path):
        """Downloads a file with a progress bar."""
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    def load_data(self):
        """Loads the dataset based on the name."""
        dataset_name = self.dataset_name.lower()
        
        # Financial datasets (10 total)
        if dataset_name == 'adult':
            self._load_adult()
        elif dataset_name == 'credit_default':
            self._load_credit_default()
        elif dataset_name == 'german_credit':
            self._load_german_credit()
        elif dataset_name == 'bank_marketing':
            self._load_bank_marketing()
        elif dataset_name == 'credit_fraud':
            self._load_credit_fraud()
        elif dataset_name == 'lending_club':
            self._load_lending_club()
        elif dataset_name == 'credit_approval':
            self._load_credit_approval()
        elif dataset_name == 'give_me_credit':
            self._load_give_me_credit()
        elif dataset_name == 'polish_bankruptcy':
            self._load_polish_bankruptcy()
        elif dataset_name == 'australian_credit':
            self._load_australian_credit()
        # Medical datasets
        elif dataset_name == 'heart_disease':
            self._load_heart_disease()
        elif dataset_name == 'diabetes':
            self._load_diabetes()
        elif dataset_name == 'breast_cancer':
            self._load_breast_cancer()
        # General imbalanced datasets
        elif dataset_name == 'ionosphere':
            self._load_ionosphere()
        elif dataset_name == 'wine_quality':
            self._load_wine_quality()
        elif dataset_name == 'covertype':
            self._load_covertype()
        elif dataset_name == 'shoppers':
            self._load_shoppers()
        elif dataset_name == 'magic':
            self._load_magic()
        elif dataset_name == 'default_payment':
            self._load_default_payment()
        else:
            available = "FINANCIAL: adult, credit_default, german_credit, bank_marketing, credit_fraud, lending_club, credit_approval, give_me_credit, polish_bankruptcy, australian_credit | MEDICAL: heart_disease, diabetes, breast_cancer | OTHER: ionosphere, wine_quality, covertype, shoppers, magic"
            raise ValueError(f"Dataset {self.dataset_name} not supported. Available: {available}")

    def _load_adult(self):
        """Load Adult/Census Income dataset from UCI."""
        self.target_col = 'salary'
        self.categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'sex', 'salary']
        self.numerical_cols = ['age', 'hours-per-week']
        
        cache_path = os.path.join(self.cache_dir, 'adult.csv')
        if os.path.exists(cache_path):
            print(f"Loading cached Adult dataset from {cache_path}...")
            self.data = pd.read_csv(cache_path)
        else:
            columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
                       'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
                       'hours-per-week', 'native-country', 'salary']
            
            # Try multiple sources
            urls = [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
            ]
            
            temp_path = os.path.join(self.cache_dir, 'temp_adult_raw.csv')
            
            for i, url in enumerate(urls):
                try:
                    self._download_with_progress(url, temp_path)
                    
                    if 'jbrownlee' in url:
                        self.data = pd.read_csv(temp_path, header=None, names=columns, na_values='?', skipinitialspace=True)
                    else:
                        self.data = pd.read_csv(temp_path, header=None, names=columns, na_values=' ?', skipinitialspace=True)
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    break
                except Exception as e:
                    print(f"Source {i+1} failed: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if i == len(urls) - 1:
                        raise
            
            self.data.dropna(inplace=True)
            
            # Select subset of columns for faster training
            keep_cols = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'sex', 'hours-per-week', 'salary']
            self.data = self.data[keep_cols]
            
            # Save to cache
            print(f"Caching Adult dataset to {cache_path}...")
            self.data.to_csv(cache_path, index=False)
        
        # Cast categorical columns to string dtype
        for col in self.categorical_cols:
            self.data[col] = self.data[col].astype(str)
            
        print(f"Loaded Adult dataset with shape {self.data.shape}")

    def _load_credit_default(self):
        """Load Taiwan Credit Card Default dataset from UCI."""
        self.target_col = 'default'
        cache_path = os.path.join(self.cache_dir, 'credit_default.csv')
        
        if os.path.exists(cache_path):
            print(f"Loading cached Credit Default dataset from {cache_path}...")
            self.data = pd.read_csv(cache_path)
        else:
            # Multiple sources for reliability
            urls = [
                ("https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls", "excel"),
                ("https://raw.githubusercontent.com/gastonstat/CreditScoring/master/CreditScoring.csv", "csv"),
            ]
            
            loaded = False
            temp_path = os.path.join(self.cache_dir, 'temp_credit_default_raw')
            
            for i, (url, file_type) in enumerate(urls):
                try:
                    self._download_with_progress(url, temp_path)
                    
                    if file_type == "excel":
                        self.data = pd.read_excel(temp_path, header=1)
                    else:
                        self.data = pd.read_csv(temp_path)
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    loaded = True
                    break
                except Exception as e:
                    print(f"Source {i+1} failed: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            if not loaded:
                raise FileNotFoundError("Failed to load Credit Default dataset from all sources.")
            
            # Handle different target column names
            target_candidates = ['default payment next month', 'default.payment.next.month', 'default', 'Status']
            for candidate in target_candidates:
                if candidate in self.data.columns:
                    if candidate != 'default':
                        self.data = self.data.rename(columns={candidate: 'default'})
                    break
            
            # If 'Status' column exists (from CreditScoring), convert to binary default
            if 'Status' in self.data.columns or 'default' not in self.data.columns:
                # Find binary target column
                for col in self.data.columns:
                    unique_vals = self.data[col].nunique()
                    if unique_vals == 2 and col not in ['SEX', 'MARRIAGE']:
                        self.data = self.data.rename(columns={col: 'default'})
                        break
            
            # Ensure default column exists
            if 'default' not in self.data.columns:
                # Create a synthetic target based on some heuristic
                print("Warning: No target column found. Creating synthetic target.")
                self.data['default'] = (self.data.iloc[:, -1] > self.data.iloc[:, -1].median()).astype(int)
            
            # Drop ID column if present
            if 'ID' in self.data.columns:
                self.data = self.data.drop(columns=['ID'])
            
            # Save to cache
            print(f"Caching Credit Default dataset to {cache_path}...")
            self.data.to_csv(cache_path, index=False)
            
        # Dynamically determine categorical columns (low cardinality)
        self.categorical_cols = ['default']
        for col in self.data.columns:
            if col != 'default' and self.data[col].nunique() < 15:
                self.categorical_cols.append(col)
        
        self.numerical_cols = [col for col in self.data.columns if col not in self.categorical_cols]
        
        # Cast categorical columns to string dtype
        for col in self.categorical_cols:
            self.data[col] = self.data[col].astype(str)
            
        print(f"Loaded Credit Default dataset with shape {self.data.shape}")

    def _load_german_credit(self):
        """Load German Credit dataset from UCI."""
        self.target_col = 'class'
        self.categorical_cols = ['status', 'history', 'purpose', 'savings', 'employment', 
                                  'personal_status', 'debtors', 'property', 'plans', 
                                  'housing', 'job', 'telephone', 'foreign', 'class']
        self.numerical_cols = ['duration', 'amount', 'installment_rate', 'residence', 
                               'age', 'credits', 'dependents']
                               
        cache_path = os.path.join(self.cache_dir, 'german_credit.csv')
        
        if os.path.exists(cache_path):
            print(f"Loading cached German Credit dataset from {cache_path}...")
            self.data = pd.read_csv(cache_path)
        else:
            columns = ['status', 'duration', 'history', 'purpose', 'amount', 'savings', 
                       'employment', 'installment_rate', 'personal_status', 'debtors', 
                       'residence', 'property', 'age', 'plans', 'housing', 'credits', 
                       'job', 'dependents', 'telephone', 'foreign', 'class']
            
            urls = [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
                "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv"
            ]
            
            temp_path = os.path.join(self.cache_dir, 'temp_german_credit_raw.csv')
            
            for i, url in enumerate(urls):
                try:
                    self._download_with_progress(url, temp_path)
                    
                    if 'jbrownlee' in url:
                        self.data = pd.read_csv(temp_path, header=None, names=columns)
                    else:
                        self.data = pd.read_csv(temp_path, header=None, names=columns, sep=' ')
                        
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    break
                except Exception as e:
                    print(f"Source {i+1} failed: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if i == len(urls) - 1:
                        raise
            
            # Convert class: 1=Good, 2=Bad -> 0=Good, 1=Bad (minority)
            self.data['class'] = self.data['class'] - 1
            
            # Save to cache
            print(f"Caching German Credit dataset to {cache_path}...")
            self.data.to_csv(cache_path, index=False)
            
        # Cast categorical columns to string dtype
        for col in self.categorical_cols:
            self.data[col] = self.data[col].astype(str)
            
        print(f"Loaded German Credit dataset with shape {self.data.shape}")

    def _load_bank_marketing(self):
        """Load Bank Marketing dataset from UCI."""
        self.target_col = 'y'
        self.categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                                  'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']
                                  
        cache_path = os.path.join(self.cache_dir, 'bank_marketing.csv')
        
        if os.path.exists(cache_path):
            print(f"Loading cached Bank Marketing dataset from {cache_path}...")
            self.data = pd.read_csv(cache_path)
        else:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
            temp_path = os.path.join(self.cache_dir, 'temp_bank.zip')
            
            import io
            import zipfile
            
            try:
                self._download_with_progress(url, temp_path)
                
                with zipfile.ZipFile(temp_path) as z:
                    # Use the full dataset
                    with z.open('bank-additional/bank-additional-full.csv') as f:
                        self.data = pd.read_csv(f, sep=';')
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                # Fallback to alternative source
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                alt_url = "https://raw.githubusercontent.com/selva86/datasets/master/bank-additional-full.csv"
                print(f"Primary URL failed, trying alternative...")
                
                temp_csv = os.path.join(self.cache_dir, 'temp_bank_alt.csv')
                self._download_with_progress(alt_url, temp_csv)
                self.data = pd.read_csv(temp_csv, sep=';')
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)
            
            # Convert target: yes/no -> 1/0
            self.data['y'] = (self.data['y'] == 'yes').astype(int)
            
            # Save to cache
            print(f"Caching Bank Marketing dataset to {cache_path}...")
            self.data.to_csv(cache_path, index=False)
            
        self.numerical_cols = [col for col in self.data.columns if col not in self.categorical_cols]
        
        # Cast categorical columns to string dtype
        for col in self.categorical_cols:
            self.data[col] = self.data[col].astype(str)
            
        print(f"Loaded Bank Marketing dataset with shape {self.data.shape}")

    def _load_credit_fraud(self):
        """Load Credit Card Fraud dataset. 
        Note: This requires the file to be downloaded from Kaggle and placed in datasets/credit_fraud/"""
        import os
        
        # Check for local file first
        local_path = os.path.join(os.path.dirname(__file__), 'datasets', 'credit_fraud', 'creditcard.csv')
        
        if os.path.exists(local_path):
            print(f"Loading Credit Fraud dataset from local file...")
            self.data = pd.read_csv(local_path)
        else:
            # Try alternative source
            print(f"Credit Fraud dataset not found locally.")
            print(f"Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            print(f"And place creditcard.csv in: {os.path.dirname(local_path)}")
            
            # For now, try a GitHub mirror
            alt_url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
            print(f"Attempting to download from alternative source...")
            try:
                self.data = pd.read_csv(alt_url)
            except Exception:
                raise FileNotFoundError(
                    f"Credit Fraud dataset not available. Please download from Kaggle and place in {local_path}"
                )
        
        # Sample if dataset is too large (284k rows)
        if len(self.data) > 50000:
            print(f"Sampling dataset from {len(self.data)} to 50,000 rows for faster training...")
            # Stratified sampling to preserve fraud ratio
            fraud = self.data[self.data['Class'] == 1]
            normal = self.data[self.data['Class'] == 0].sample(n=50000-len(fraud), random_state=self.random_state)
            self.data = pd.concat([fraud, normal]).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        self.target_col = 'Class'
        self.categorical_cols = ['Class']  # Only target is categorical
        self.numerical_cols = [col for col in self.data.columns if col != 'Class']
        print(f"Loaded Credit Fraud dataset with shape {self.data.shape}")

        print(f"Loaded {self.dataset_name} dataset with shape {self.data.shape}")

    def preprocess(self):
        """Encodes categorical variables and scales numerical ones."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.data.copy()

        # Encode categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Scale numerical columns
        if self.numerical_cols:
            df[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])

        self.data = df
        print("Data preprocessed.")

    def get_rare_event_stats(self):
        """Identifies the minority class and its prevalence."""
        if self.target_col not in self.data.columns:
             return None
        
        counts = self.data[self.target_col].value_counts(normalize=True)
        minority_class = counts.idxmin()
        minority_ratio = counts.min()
        
        print(f"Target: {self.target_col}")
        print(f"Minority Class: {minority_class} (Ratio: {minority_ratio:.4f})")
        
        return {
            'target': self.target_col,
            'minority_class': minority_class,
            'minority_ratio': minority_ratio
        }

    def split_data(self, test_size=0.2):
        """Splits data into train and test sets."""
        X = self.data.drop(columns=[self.target_col]) if self.target_col else self.data
        y = self.data[self.target_col] if self.target_col else None
        
        if y is not None:
             X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
             # Recombine for generative models which usually take the full row
             train_df = pd.concat([X_train, y_train], axis=1)
             test_df = pd.concat([X_test, y_test], axis=1)
        else:
             train_df, test_df = train_test_split(self.data, test_size=test_size, random_state=self.random_state)
             
        return train_df, test_df

    # ============== NEW DATASET LOADERS ==============
    
    def _load_heart_disease(self):
        """Load Heart Disease dataset from UCI."""
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
            "https://raw.githubusercontent.com/selva86/datasets/master/heart.csv"
        ]
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        loaded = False
        for i, url in enumerate(urls):
            try:
                print(f"Downloading Heart Disease dataset (attempt {i+1})...")
                if 'selva86' in url:
                    self.data = pd.read_csv(url)
                else:
                    self.data = pd.read_csv(url, header=None, names=columns, na_values='?')
                loaded = True
                break
            except Exception as e:
                print(f"Source {i+1} failed: {e}")
        
        if not loaded:
            raise FileNotFoundError("Failed to load Heart Disease dataset from all sources.")
        
        self.data.dropna(inplace=True)
        # Convert target: 0 = no disease, 1-4 = disease -> binary
        if 'target' in self.data.columns and self.data['target'].max() > 1:
            self.data['target'] = (self.data['target'] > 0).astype(int)
        
        self.target_col = 'target'
        self.categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'target']
        self.numerical_cols = [col for col in self.data.columns if col not in self.categorical_cols]
        print(f"Loaded Heart Disease dataset with shape {self.data.shape}")

    def _load_diabetes(self):
        """Load Pima Indians Diabetes dataset."""
        urls = [
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        ]
        columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                   'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome']
        
        for i, url in enumerate(urls):
            try:
                print(f"Downloading Diabetes dataset (attempt {i+1})...")
                self.data = pd.read_csv(url, header=None, names=columns)
                break
            except Exception as e:
                print(f"Source {i+1} failed: {e}")
                if i == len(urls) - 1:
                    raise
        
        self.target_col = 'outcome'
        self.categorical_cols = ['outcome']
        self.numerical_cols = [col for col in self.data.columns if col != 'outcome']
        print(f"Loaded Diabetes dataset with shape {self.data.shape}")

    def _load_breast_cancer(self):
        """Load Breast Cancer Wisconsin dataset."""
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.csv"
        ]
        
        for i, url in enumerate(urls):
            try:
                print(f"Downloading Breast Cancer dataset (attempt {i+1})...")
                if 'wdbc' in url:
                    columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(30)]
                    self.data = pd.read_csv(url, header=None, names=columns)
                    self.data = self.data.drop(columns=['id'])
                    self.data['diagnosis'] = (self.data['diagnosis'] == 'M').astype(int)
                else:
                    self.data = pd.read_csv(url, header=None)
                    self.data.columns = [f'feature_{i}' for i in range(len(self.data.columns)-1)] + ['diagnosis']
                break
            except Exception as e:
                print(f"Source {i+1} failed: {e}")
                if i == len(urls) - 1:
                    raise
        
        self.target_col = 'diagnosis'
        self.categorical_cols = ['diagnosis']
        self.numerical_cols = [col for col in self.data.columns if col != 'diagnosis']
        print(f"Loaded Breast Cancer dataset with shape {self.data.shape}")

    def _load_ionosphere(self):
        """Load Ionosphere dataset from UCI."""
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv"
        ]
        
        for i, url in enumerate(urls):
            try:
                print(f"Downloading Ionosphere dataset (attempt {i+1})...")
                self.data = pd.read_csv(url, header=None)
                break
            except Exception as e:
                print(f"Source {i+1} failed: {e}")
                if i == len(urls) - 1:
                    raise
        
        # Last column is target (g=good, b=bad)
        self.data.columns = [f'feature_{i}' for i in range(len(self.data.columns)-1)] + ['target']
        self.data['target'] = (self.data['target'] == 'b').astype(int)
        
        self.target_col = 'target'
        self.categorical_cols = ['target']
        self.numerical_cols = [col for col in self.data.columns if col != 'target']
        print(f"Loaded Ionosphere dataset with shape {self.data.shape}")

    def _load_wine_quality(self):
        """Load Wine Quality dataset from UCI."""
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "https://raw.githubusercontent.com/selva86/datasets/master/winequality-red.csv"
        ]
        
        loaded = False
        for i, url in enumerate(urls):
            try:
                print(f"Downloading Wine Quality dataset (attempt {i+1})...")
                self.data = pd.read_csv(url, sep=';')
                loaded = True
                break
            except Exception as e:
                print(f"Source {i+1} failed: {e}")
        
        if not loaded:
            raise FileNotFoundError("Failed to load Wine Quality dataset from all sources.")
        
        # Convert quality to binary: 0-5 = low, 6-10 = high - only if not already binary
        if self.data['quality'].max() > 1:
            self.data['quality'] = (self.data['quality'] >= 7).astype(int)
        
        self.target_col = 'quality'
        self.categorical_cols = ['quality']
        self.numerical_cols = [col for col in self.data.columns if col != 'quality']
        print(f"Loaded Wine Quality dataset with shape {self.data.shape}")

    def _load_covertype(self):
        """Load Covertype dataset from UCI (sampled)."""
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        ]
        
        try:
            print(f"Downloading Covertype dataset...")
            self.data = pd.read_csv(urls[0], header=None, compression='gzip')
        except Exception as e:
            raise FileNotFoundError(f"Failed to load Covertype dataset: {e}")
        
        if 'cover_type' not in self.data.columns:
            self.data.columns = [f'feature_{i}' for i in range(len(self.data.columns)-1)] + ['cover_type']
        
        # Sample if too large
        if len(self.data) > 20000:
            self.data = self.data.sample(n=20000, random_state=self.random_state)
        
        # Binary: Cover type 4 (rarest) vs others
        self.data['cover_type'] = (self.data['cover_type'] == 4).astype(int)
        
        self.target_col = 'cover_type'
        self.categorical_cols = ['cover_type']
        self.numerical_cols = [col for col in self.data.columns if col != 'cover_type']
        print(f"Loaded Covertype dataset with shape {self.data.shape}")

    def _load_shoppers(self):
        """Load Online Shoppers Intention dataset from UCI."""
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
        ]
        
        for i, url in enumerate(urls):
            try:
                print(f"Downloading Shoppers dataset (attempt {i+1})...")
                self.data = pd.read_csv(url)
                break
            except Exception as e:
                print(f"Source {i+1} failed: {e}")
                if i == len(urls) - 1:
                    raise FileNotFoundError("Failed to load Shoppers dataset from all sources.")
        
        # Convert Revenue to int
        if 'Revenue' in self.data.columns:
            self.data['Revenue'] = self.data['Revenue'].astype(int)
        
        self.target_col = 'Revenue'
        self.categorical_cols = ['Revenue']
        if 'Month' in self.data.columns:
            self.categorical_cols.extend(['Month', 'VisitorType', 'Weekend'])
        self.numerical_cols = [col for col in self.data.columns if col not in self.categorical_cols]
        print(f"Loaded Shoppers dataset with shape {self.data.shape}")

    def _load_magic(self):
        """Load MAGIC Gamma Telescope dataset from UCI."""
        urls = [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
        ]
        columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 
                   'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
        
        loaded = False
        for i, url in enumerate(urls):
            try:
                print(f"Downloading MAGIC dataset (attempt {i+1})...")
                self.data = pd.read_csv(url, header=None, names=columns)
                loaded = True
                break
            except Exception as e:
                print(f"Source {i+1} failed: {e}")
        
        if not loaded:
            raise FileNotFoundError("Failed to load MAGIC dataset from all sources.")
        
        # Convert class: g=gamma (signal), h=hadron (background) - only if string
        if self.data['class'].dtype == object:
            self.data['class'] = (self.data['class'] == 'g').astype(int)
        
        self.target_col = 'class'
        self.categorical_cols = ['class']
        self.numerical_cols = [col for col in self.data.columns if col != 'class']
        print(f"Loaded MAGIC dataset with shape {self.data.shape}")

    def _load_default_payment(self):
        """Load Default of Credit Card Clients (same as credit_default, alias)."""
        self._load_credit_default()

    # ============== ADDITIONAL FINANCIAL DATASETS ==============

    def _load_lending_club(self):
        """Load Lending Club loan data."""
        # This requires manual download usually, but we should not fake it.
        # Check for local file
        import os
        local_path = os.path.join(os.path.dirname(__file__), 'datasets', 'lending_club', 'loan.csv')
        
        if os.path.exists(local_path):
             self.data = pd.read_csv(local_path)
             # Basic preprocessing for Lending Club would go here
             # For now, if it exists, load it. If not, raise.
             self.target_col = 'loan_status' # Assuming standard LC schema
        else:
             raise FileNotFoundError(f"Lending Club dataset not found at {local_path}. Please download from Kaggle.")
             
        # Just to keep the function signature valid if we loaded data
        if self.data is not None:
             self.categorical_cols = [c for c in self.data.columns if self.data[c].dtype == 'object']
             self.numerical_cols = [c for c in self.data.columns if c not in self.categorical_cols]

    def _load_credit_approval(self):
        """Load Credit Approval dataset from UCI."""
        self.target_col = 'class'
        self.categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13', 'class']
        
        cache_path = os.path.join(self.cache_dir, 'credit_approval.csv')
        
        if os.path.exists(cache_path):
            print(f"Loading cached Credit Approval dataset from {cache_path}...")
            self.data = pd.read_csv(cache_path)
        else:
            urls = [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
            ]
            columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 
                       'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class']
            
            loaded = False
            temp_path = os.path.join(self.cache_dir, 'temp_credit_approval_raw.csv')
            
            for i, url in enumerate(urls):
                try:
                    self._download_with_progress(url, temp_path)
                    self.data = pd.read_csv(temp_path, header=None, names=columns, na_values='?')
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    loaded = True
                    break
                except Exception as e:
                    print(f"Source {i+1} failed: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            if not loaded:
                raise FileNotFoundError("Failed to load Credit Approval dataset from all sources.")
            
            self.data.dropna(inplace=True)
            # Convert class to binary
            if self.data['class'].dtype == object:
                self.data['class'] = (self.data['class'] == '+').astype(int)
            
            # Save to cache
            print(f"Caching Credit Approval dataset to {cache_path}...")
            self.data.to_csv(cache_path, index=False)
            
        self.numerical_cols = [col for col in self.data.columns if col not in self.categorical_cols]
        print(f"Loaded Credit Approval dataset with shape {self.data.shape}")

    def _load_give_me_credit(self):
        """Load Give Me Some Credit dataset (Kaggle)."""
        import os
        local_path = os.path.join(os.path.dirname(__file__), 'datasets', 'give_me_credit', 'cs-training.csv')
        
        if os.path.exists(local_path):
            self.data = pd.read_csv(local_path)
            self.target_col = 'SeriousDlqin2yrs'
            self.data = self.data.dropna()
        else:
            raise FileNotFoundError(f"Give Me Some Credit dataset not found at {local_path}. Please download from Kaggle.")
            
        self.categorical_cols = [self.target_col]
        self.numerical_cols = [col for col in self.data.columns if col != self.target_col]
        print(f"Loaded Give Me Credit dataset with shape {self.data.shape}")

    def _load_polish_bankruptcy(self):
        """Load Polish Companies Bankruptcy dataset."""
        # Requires local file
        import os
        # There are 5 files usually (1year, 2year...), picking 3rd year as example
        local_path = os.path.join(os.path.dirname(__file__), 'datasets', 'polish_bankruptcy', '3year.arff')
        
        if os.path.exists(local_path):
            from scipy.io import arff
            data, meta = arff.loadarff(local_path)
            self.data = pd.DataFrame(data)
            self.data['class'] = self.data['class'].astype(int) # Usually loaded as bytes
            self.target_col = 'class'
            
            # Handle missing values (common in this dataset)
            # Fill with 0 as NaNs often imply missing financial records or division by zero
            self.data.fillna(0, inplace=True)
        else:
             raise FileNotFoundError(f"Polish Bankruptcy dataset not found at {local_path}. Please download from UCI.")
        
        self.categorical_cols = ['class']
        self.numerical_cols = [col for col in self.data.columns if col != 'class']
        print(f"Loaded Polish Bankruptcy dataset with shape {self.data.shape}")

    def _load_australian_credit(self):
        """Load Australian Credit Approval dataset from UCI."""
        self.target_col = 'class'
        self.categorical_cols = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12', 'class']
        
        cache_path = os.path.join(self.cache_dir, 'australian_credit.csv')
        
        if os.path.exists(cache_path):
            print(f"Loading cached Australian Credit dataset from {cache_path}...")
            self.data = pd.read_csv(cache_path)
        else:
            urls = [
                "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
            ]
            
            loaded = False
            temp_path = os.path.join(self.cache_dir, 'temp_aus_credit_raw.csv')
            
            for i, url in enumerate(urls):
                try:
                    self._download_with_progress(url, temp_path)
                    self.data = pd.read_csv(temp_path, header=None, sep=' ')
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    loaded = True
                    break
                except Exception as e:
                    print(f"Source {i+1} failed: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            if not loaded:
                raise FileNotFoundError("Failed to load Australian Credit dataset from all sources.")
            
            # Assign column names if loaded from UCI
            if self.data.shape[1] == 15:
                self.data.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 
                                     'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'class']
            
            # Save to cache
            print(f"Caching Australian Credit dataset to {cache_path}...")
            self.data.to_csv(cache_path, index=False)
        
        self.numerical_cols = [col for col in self.data.columns if col not in self.categorical_cols]
        
        # Cast categorical columns to string dtype
        for col in self.categorical_cols:
            self.data[col] = self.data[col].astype(str)
            
        print(f"Loaded Australian Credit dataset with shape {self.data.shape}")


if __name__ == "__main__":
    # Test all financial datasets
    datasets = ['adult', 'credit_default', 'german_credit', 'bank_marketing', 
                'lending_club', 'credit_approval', 'give_me_credit', 
                'polish_bankruptcy', 'australian_credit']
    
    print("="*70)
    print("FINANCIAL DATASETS TEST")
    print("="*70)
    
    for ds in datasets:
        print(f"\n--- Testing {ds} ---")
        try:
            loader = DataLoader(ds)
            loader.load_data()
            stats = loader.get_rare_event_stats()
            print(f"SUCCESS: {loader.data.shape}, Minority: {stats['minority_ratio']:.2%}")
        except Exception as e:
            print(f"FAILED: {e}")
