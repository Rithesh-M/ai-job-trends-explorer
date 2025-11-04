"""
AI-Based Job Recommendation System
Uses TF-IDF and Cosine Similarity for job recommendations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
import os
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, OSError):
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download punkt: {e}")

try:
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, OSError):
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        pass  # punkt_tab may not be available in all NLTK versions
    
try:
    nltk.data.find('corpora/stopwords')
except (LookupError, OSError):
    try:
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download stopwords: {e}")

class JobRecommender:
    def __init__(self, csv_path=None):
        """Initialize recommender with dataset"""
        print("🤖 Initializing AI Job Recommender...")
        
        # Use absolute path if not provided
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned_linkedin_jobs.csv')
        
        self.df = pd.read_csv(csv_path)
        self.df = self.df.head(5000)  # Limit for performance
        self.vectorizer = None
        self.tfidf_matrix = None
        
        # Handle stopwords with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: Could not load stopwords, using empty set: {e}")
            self.stop_words = set()
        
        print(f"✓ Loaded {len(self.df)} jobs for recommendation engine")
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def prepare_features(self):
        """Prepare job features for TF-IDF"""
        print("📝 Preparing job features...")
        
        # Combine job title and details for better matching
        self.df['combined_features'] = (
            self.df['job'].astype(str) + ' ' + 
            self.df['job_details'].astype(str).str[:500]  # Limit description length
        )
        
        # Preprocess
        self.df['processed_features'] = self.df['combined_features'].apply(self.preprocess_text)
        
        print("✓ Features prepared")
    
    def train_model(self):
        """Train TF-IDF model"""
        print("🎓 Training TF-IDF model...")
        
        self.prepare_features()
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_features'])
        
        print(f"✓ Model trained with {self.tfidf_matrix.shape[1]} features")
        
        # Save model
        self.save_model()
    
    def get_recommendations(self, query, top_n=10, location_filter=None, work_type_filter=None):
        """Get job recommendations based on query"""
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Transform query
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top indices
        top_indices = similarities.argsort()[-top_n*3:][::-1]  # Get more for filtering
        
        # Filter results
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        
        # Apply filters
        if location_filter and location_filter != 'all':
            results = results[results['location'].str.contains(location_filter, case=False, na=False)]
        
        if work_type_filter and work_type_filter != 'all':
            results = results[results['work_type'].str.contains(work_type_filter, case=False, na=False)]
        
        # Return top N after filtering
        results = results.head(top_n)
        
        return results[[
            'job', 'company_name', 'location', 'work_type', 
            'no_of_application', 'linkedin_followers', 'posted_hours_ago',
            'job_details', 'similarity_score'
        ]].to_dict('records')
    
    def cluster_jobs(self, n_clusters=8):
        """Cluster jobs using KMeans"""
        print(f"🔍 Clustering jobs into {n_clusters} categories...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(self.tfidf_matrix)
        
        # Get top terms per cluster
        cluster_info = {}
        feature_names = self.vectorizer.get_feature_names_out()
        
        for i in range(n_clusters):
            cluster_center = kmeans.cluster_centers_[i]
            top_indices = cluster_center.argsort()[-10:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            
            cluster_jobs = self.df[self.df['cluster'] == i]
            cluster_info[f'Cluster {i+1}'] = {
                'top_terms': top_terms,
                'job_count': len(cluster_jobs),
                'sample_jobs': cluster_jobs['job'].head(5).tolist()
            }
        
        print("✓ Clustering complete")
        return cluster_info
    
    def extract_top_skills(self, top_n=50):
        """Extract most frequent skills/keywords"""
        print("💡 Extracting top skills...")
        
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = self.tfidf_matrix.sum(axis=0).A1
        
        # Get top features
        top_indices = tfidf_scores.argsort()[-top_n:][::-1]
        top_skills = [(feature_names[i], tfidf_scores[i]) for i in top_indices]
        
        print("✓ Skills extracted")
        return top_skills
    
    def save_model(self):
        """Save trained model"""
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        joblib.dump(self.vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
        joblib.dump(self.tfidf_matrix, os.path.join(models_dir, 'tfidf_matrix.pkl'))
        self.df.to_pickle(os.path.join(models_dir, 'processed_jobs.pkl'))
        
        print("✓ Model saved to 'models/' directory")
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
            self.vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
            self.tfidf_matrix = joblib.load(os.path.join(models_dir, 'tfidf_matrix.pkl'))
            self.df = pd.read_pickle(os.path.join(models_dir, 'processed_jobs.pkl'))
            print("✓ Model loaded successfully")
            return True
        except:
            print("⚠️ No saved model found. Training new model...")
            return False

if __name__ == "__main__":
    # Initialize and train
    recommender = JobRecommender()
    recommender.train_model()
    
    # Test clustering
    cluster_info = recommender.cluster_jobs()
    
    # Extract skills
    top_skills = recommender.extract_top_skills()
    
    print("\n🎯 Top 10 Skills/Keywords:")
    for skill, score in top_skills[:10]:
        print(f"  • {skill}: {score:.2f}")
    
    # Test recommendations
    print("\n🔍 Testing recommendations for 'data analyst':")
    recommendations = recommender.get_recommendations('data analyst', top_n=5)
    for i, job in enumerate(recommendations, 1):
        print(f"\n{i}. {job['job']}")
        print(f"   Company: {job['company_name']}")
        print(f"   Location: {job['location']}")
        print(f"   Similarity: {job['similarity_score']:.3f}")
    
    print("\n✅ AI Recommender ready!")