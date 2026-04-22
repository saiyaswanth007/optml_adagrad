import random
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp

def set_seeds(seed=42):
    """Ensure absolute reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class SparseDataset(Dataset):
    def __init__(self, X, y):
        """
        X: scipy.sparse.csr_matrix
        y: numpy array
        """
        # Precompute massive dense tensor once to avoid per-item overhead
        self.X_dense = torch.FloatTensor(X.toarray())
        self.y_tensor = torch.LongTensor(y)

    def __len__(self):
        return self.X_dense.shape[0]

    def __getitem__(self, idx):
        return self.X_dense[idx], self.y_tensor[idx]

def get_dataloaders(max_features=50000, batch_size=64, seed=42):
    set_seeds(seed)
    
    print("Fetching 20 Newsgroups dataset...")
    # Using predefined train/test split from sklearn
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.95,
        min_df=5,
        ngram_range=(1,1),
        max_features=max_features
    )
    X_train = vectorizer.fit_transform(newsgroups_train.data)
    X_test = vectorizer.transform(newsgroups_test.data)
    
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target
    
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    train_dataset = SparseDataset(X_train, y_train)
    test_dataset = SparseDataset(X_test, y_test)
    
    # shuffle=False by default here? No, shuffling is typical, but seed is set.
    # To assure IDENTICAL batching, we will shuffle but seed guarantees same order.
    # Wait, the instruction says "identical batching order". 
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, vectorizer
