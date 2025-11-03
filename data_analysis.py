"""
Data Analysis & Preprocessing Module
Analyzes LinkedIn job dataset and generates insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
import os
warnings.filterwarnings('ignore')

class JobDataAnalyzer:
    def __init__(self, csv_path='../cleaned_linkedin_jobs.csv'):
        """Initialize with dataset path"""
        print("📊 Loading dataset...")
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} job records")
        
        # Ensure static/images directory exists
        self.images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
    def get_summary_stats(self):
        """Get basic dataset statistics"""
        stats = {
            'total_jobs': len(self.df),
            'total_companies': self.df['company_name'].nunique(),
            'total_locations': self.df['location'].nunique(),
            'work_types': self.df['work_type'].value_counts().to_dict(),
            'avg_applications': self.df['no_of_application'].mean(),
            'avg_followers': self.df['linkedin_followers'].mean()
        }
        return stats
    
    def analyze_top_jobs(self, top_n=15):
        """Analyze most common job titles"""
        top_jobs = self.df['job'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_jobs.values, y=top_jobs.index, palette='viridis')
        plt.title(f'Top {top_n} Job Titles', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Postings')
        plt.ylabel('Job Title')
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'top_jobs.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return top_jobs.to_dict()
    
    def analyze_top_companies(self, top_n=15):
        """Analyze top hiring companies"""
        top_companies = self.df['company_name'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_companies.values, y=top_companies.index, palette='coolwarm')
        plt.title(f'Top {top_n} Hiring Companies', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Job Postings')
        plt.ylabel('Company')
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'top_companies.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return top_companies.to_dict()
    
    def analyze_locations(self, top_n=15):
        """Analyze top job locations"""
        top_locations = self.df['location'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_locations.values, y=top_locations.index, palette='plasma')
        plt.title(f'Top {top_n} Job Locations', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Jobs')
        plt.ylabel('Location')
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'top_locations.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return top_locations.to_dict()
    
    def analyze_work_types(self):
        """Analyze work type distribution"""
        work_types = self.df['work_type'].value_counts()
        
        plt.figure(figsize=(10, 8))
        colors = sns.color_palette('Set2', len(work_types))
        plt.pie(work_types.values, labels=work_types.index, autopct='%1.1f%%',
                startangle=90, colors=colors)
        plt.title('Work Type Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'work_types.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return work_types.to_dict()
    
    def analyze_company_followers(self, top_n=15):
        """Analyze companies by LinkedIn followers"""
        company_followers = self.df.groupby('company_name')['linkedin_followers'].mean().sort_values(ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=company_followers.values, y=company_followers.index, palette='mako')
        plt.title(f'Top {top_n} Companies by LinkedIn Followers', fontsize=16, fontweight='bold')
        plt.xlabel('Average LinkedIn Followers')
        plt.ylabel('Company')
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'company_followers.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return company_followers.to_dict()
    
    def analyze_posting_time(self):
        """Analyze job posting recency"""
        plt.figure(figsize=(12, 6))
        self.df['posted_hours_ago'].hist(bins=50, color='skyblue', edgecolor='black')
        plt.title('Job Posting Recency Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Hours Ago')
        plt.ylabel('Number of Jobs')
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'posting_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        recent_jobs = len(self.df[self.df['posted_hours_ago'] <= 24])
        week_old = len(self.df[(self.df['posted_hours_ago'] > 24) & (self.df['posted_hours_ago'] <= 168)])
        
        return {'recent_24h': recent_jobs, 'week_old': week_old}
    
    def generate_job_wordcloud(self):
        """Generate word cloud from job titles"""
        text = ' '.join(self.df['job'].astype(str))
        
        wordcloud = WordCloud(width=1200, height=600, background_color='white',
                            colormap='viridis', max_words=100).generate(text)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Job Titles Word Cloud', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'job_wordcloud.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Generated job titles word cloud")
    
    def generate_skills_wordcloud(self):
        """Generate word cloud from job details/descriptions"""
        text = ' '.join(self.df['job_details'].astype(str))
        
        wordcloud = WordCloud(width=1200, height=600, background_color='white',
                            colormap='plasma', max_words=150).generate(text)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Skills & Keywords Word Cloud', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'skills_wordcloud.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Generated skills word cloud")
    
    def run_complete_analysis(self):
        """Run all analyses and generate visualizations"""
        print("\n🔍 Starting comprehensive data analysis...\n")
        
        # Get summary stats
        stats = self.get_summary_stats()
        print(f"📈 Total Jobs: {stats['total_jobs']}")
        print(f"🏢 Total Companies: {stats['total_companies']}")
        print(f"📍 Total Locations: {stats['total_locations']}\n")
        
        # Generate all visualizations
        print("📊 Analyzing top jobs...")
        self.analyze_top_jobs()
        
        print("🏢 Analyzing top companies...")
        self.analyze_top_companies()
        
        print("📍 Analyzing locations...")
        self.analyze_locations()
        
        print("💼 Analyzing work types...")
        self.analyze_work_types()
        
        print("👥 Analyzing company followers...")
        self.analyze_company_followers()
        
        print("⏰ Analyzing posting times...")
        self.analyze_posting_time()
        
        print("☁️ Generating word clouds...")
        self.generate_job_wordcloud()
        self.generate_skills_wordcloud()
        
        print("\n✅ Analysis complete! All visualizations saved in 'static/images/'\n")
        
        return stats

if __name__ == "__main__":
    analyzer = JobDataAnalyzer()
    results = analyzer.run_complete_analysis()