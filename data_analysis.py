"""
Data Analysis & Preprocessing Module
Analyzes LinkedIn job dataset and generates insights

Usage:
    python job_data_analyzer.py
"""

from pathlib import Path
from typing import Union, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
import json

warnings.filterwarnings('ignore')

class JobDataAnalyzer:
    def __init__(self, csv_path: Optional[Union[str, Path]] = None):
        """Initialize with dataset path and prepare images directory."""
        print("📊 Loading dataset...")

        # default path: same folder as this script
        base = Path(__file__).resolve().parent
        if csv_path is None:
            csv_path = base / 'cleaned_linkedin_jobs.csv'
        else:
            csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {csv_path}")

        # load dataset
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} job records from {csv_path.name}")

        # prepare images directory
        self.images_dir = base / 'static' / 'images'
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # basic cleanup / ensure expected columns exist
        self._ensure_columns()

    def _ensure_columns(self):
        """Do light validation and convert expected numeric columns to numeric types."""
        expected = ['job', 'company_name', 'location', 'work_type',
                    'no_of_application', 'linkedin_followers', 'posted_hours_ago', 'job_details']
        present = set(self.df.columns)
        missing = [c for c in expected if c not in present]
        if missing:
            # warn but don't crash; some analyses will just skip
            print(f"⚠️ Warning: Missing columns (some analyses will be limited): {missing}")

        # convert numeric-like columns safely if present
        for col in ['no_of_application', 'linkedin_followers', 'posted_hours_ago']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # fill NaNs in text fields with empty string for wordclouds
        for col in ['job', 'job_details', 'company_name', 'location', 'work_type']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('').astype(str)

    def get_summary_stats(self) -> dict:
        """Get basic dataset statistics"""
        stats = {
            'total_jobs': len(self.df),
            'total_companies': int(self.df['company_name'].nunique()) if 'company_name' in self.df.columns else None,
            'total_locations': int(self.df['location'].nunique()) if 'location' in self.df.columns else None,
            'work_types': self.df['work_type'].value_counts().to_dict() if 'work_type' in self.df.columns else {},
            'avg_applications': float(self.df['no_of_application'].mean()) if 'no_of_application' in self.df.columns else None,
            'avg_followers': float(self.df['linkedin_followers'].mean()) if 'linkedin_followers' in self.df.columns else None
        }
        return stats

    def analyze_top_jobs(self, top_n: Optional[int] = None) -> dict:
        """
        Analyze job title frequencies.

        - If top_n is None (default) -> compute & return counts for ALL job titles,
          and save a plot named 'all_jobs.png'.
        - If top_n is an int -> compute counts for all titles but plot only the top_n
          and save as 'top_jobs.png'.

        Returns the FULL counts as a dict (job_title -> count).
        """
        if 'job' not in self.df.columns:
            print("⚠️ 'job' column not present in dataset. Returning empty dict.")
            return {}

        # compute counts for all job titles (sorted descending)
        job_counts = self.df['job'].value_counts()

        # choose what to plot
        if top_n is None:
            counts_to_plot = job_counts
            filename = 'all_jobs.png'
            title = 'All Job Titles'
        else:
            counts_to_plot = job_counts.head(top_n)
            filename = 'top_jobs.png'
            title = f'Top {top_n} Job Titles'

        # dynamic figure height so many labels remain readable
        height = max(6, 0.25 * len(counts_to_plot))
        plt.figure(figsize=(12, height))

        # horizontal barplot (values on x, labels on y)
        sns.barplot(x=counts_to_plot.values, y=counts_to_plot.index, palette='viridis')

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Number of Postings')
        plt.ylabel('Job Title')

        # adjust font size for many labels
        if len(counts_to_plot) > 30:
            plt.yticks(fontsize=9)

        plt.tight_layout()
        save_path = self.images_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved job title plot to: {save_path}")
        # always return the FULL counts dict (not just the plotted subset)
        return job_counts.to_dict()

    def analyze_top_companies(self, top_n: int = 15) -> dict:
        """Analyze top hiring companies (like original)."""
        if 'company_name' not in self.df.columns:
            print("⚠️ 'company_name' not present.")
            return {}
        top_companies = self.df['company_name'].value_counts().head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_companies.values, y=top_companies.index, palette='coolwarm')
        plt.title(f'Top {top_n} Hiring Companies', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Job Postings')
        plt.ylabel('Company')
        plt.tight_layout()
        plt.savefig(self.images_dir / 'top_companies.png', dpi=300, bbox_inches='tight')
        plt.close()

        return top_companies.to_dict()

    def analyze_locations(self, top_n: int = 15) -> dict:
        """Analyze top job locations."""
        if 'location' not in self.df.columns:
            print("⚠️ 'location' not present.")
            return {}
        top_locations = self.df['location'].value_counts().head(top_n)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_locations.values, y=top_locations.index, palette='plasma')
        plt.title(f'Top {top_n} Job Locations', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Jobs')
        plt.ylabel('Location')
        plt.tight_layout()
        plt.savefig(self.images_dir / 'top_locations.png', dpi=300, bbox_inches='tight')
        plt.close()

        return top_locations.to_dict()

    def analyze_work_types(self) -> dict:
        """Analyze work type distribution."""
        if 'work_type' not in self.df.columns:
            print("⚠️ 'work_type' not present.")
            return {}
        work_types = self.df['work_type'].value_counts()

        plt.figure(figsize=(10, 8))
        colors = sns.color_palette('Set2', len(work_types))
        plt.pie(work_types.values, labels=work_types.index, autopct='%1.1f%%',
                startangle=90, colors=colors)
        plt.title('Work Type Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.images_dir / 'work_types.png', dpi=300, bbox_inches='tight')
        plt.close()

        return work_types.to_dict()

    def analyze_company_followers(self, top_n: int = 15) -> dict:
        """Analyze companies by LinkedIn followers (average)."""
        if 'company_name' not in self.df.columns or 'linkedin_followers' not in self.df.columns:
            print("⚠️ Required columns for company followers analysis are missing.")
            return {}
        company_followers = (self.df.groupby('company_name')['linkedin_followers']
                             .mean().sort_values(ascending=False).head(top_n))

        plt.figure(figsize=(12, 6))
        sns.barplot(x=company_followers.values, y=company_followers.index, palette='mako')
        plt.title(f'Top {top_n} Companies by LinkedIn Followers', fontsize=16, fontweight='bold')
        plt.xlabel('Average LinkedIn Followers')
        plt.ylabel('Company')
        plt.tight_layout()
        plt.savefig(self.images_dir / 'company_followers.png', dpi=300, bbox_inches='tight')
        plt.close()

        return company_followers.to_dict()

    def analyze_posting_time(self) -> dict:
        """Analyze job posting recency (counts for recent vs week-old)."""
        if 'posted_hours_ago' not in self.df.columns:
            print("⚠️ 'posted_hours_ago' not present.")
            return {}

        plt.figure(figsize=(12, 6))
        self.df['posted_hours_ago'].dropna().hist(bins=50, edgecolor='black')
        plt.title('Job Posting Recency Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Hours Ago')
        plt.ylabel('Number of Jobs')
        plt.tight_layout()
        plt.savefig(self.images_dir / 'posting_time.png', dpi=300, bbox_inches='tight')
        plt.close()

        recent_jobs = int(self.df[self.df['posted_hours_ago'] <= 24].shape[0])
        week_old = int(self.df[(self.df['posted_hours_ago'] > 24) & (self.df['posted_hours_ago'] <= 168)].shape[0])

        return {'recent_24h': recent_jobs, 'week_old': week_old}

    def generate_job_wordcloud(self):
        """Generate word cloud from job titles."""
        if 'job' not in self.df.columns:
            print("⚠️ 'job' column missing; skipping wordcloud.")
            return
        text = ' '.join(self.df['job'].astype(str))
        wordcloud = WordCloud(width=1200, height=600, background_color='white',
                              colormap='viridis', max_words=100).generate(text)

        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Job Titles Word Cloud', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.images_dir / 'job_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Generated job titles word cloud")

    def generate_skills_wordcloud(self):
        """Generate word cloud from job details/descriptions."""
        if 'job_details' not in self.df.columns:
            print("⚠️ 'job_details' missing; skipping skills wordcloud.")
            return
        text = ' '.join(self.df['job_details'].astype(str))
        wordcloud = WordCloud(width=1200, height=600, background_color='white',
                              colormap='plasma', max_words=150).generate(text)

        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Skills & Keywords Word Cloud', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.images_dir / 'skills_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Generated skills word cloud")

    def save_stats(self, stats: dict, filename: str = 'summary_stats.json') -> Path:
        """Save a stats dict as JSON into static/images (or other specified)."""
        out = self.images_dir / filename
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved stats to {out}")
        return out

    def run_complete_analysis(self) -> dict:
        """Run all analyses and generate visualizations; returns summary stats."""
        print("\n🔍 Starting comprehensive data analysis...\n")

        # Get summary stats
        stats = self.get_summary_stats()
        print(f"📈 Total Jobs: {stats['total_jobs']}")
        print(f"🏢 Total Companies: {stats['total_companies']}")
        print(f"📍 Total Locations: {stats['total_locations']}\n")

        # Generate all visualizations
        print("📊 Analyzing job titles (all)...")
        stats['job_title_counts'] = self.analyze_top_jobs(top_n=None)  # returns full dict, also saves all_jobs.png

        print("🏢 Analyzing top companies...")
        stats['top_companies'] = self.analyze_top_companies()

        print("📍 Analyzing locations...")
        stats['top_locations'] = self.analyze_locations()

        print("💼 Analyzing work types...")
        stats['work_type_distribution'] = self.analyze_work_types()

        print("👥 Analyzing company followers...")
        stats['company_followers'] = self.analyze_company_followers()

        print("⏰ Analyzing posting times...")
        stats['posting_time_summary'] = self.analyze_posting_time()

        print("☁️ Generating word clouds...")
        self.generate_job_wordcloud()
        self.generate_skills_wordcloud()

        # Save stats for easy consumption by other tools
        self.save_stats(stats)

        print("\n✅ Analysis complete! All visualizations saved in 'static/images/'\n")
        return stats


if __name__ == "__main__":
    analyzer = JobDataAnalyzer()
    results = analyzer.run_complete_analysis()
