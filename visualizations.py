"""
Interactive Visualization Dashboard Module
Creates Plotly visualizations for the web app
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

class JobVisualizer:
    def __init__(self, csv_path=None):
        """Initialize with dataset"""
        # Use absolute path if not provided
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned_linkedin_jobs.csv')
        
        self.df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(self.df)} jobs for visualization")
    
    def create_top_jobs_chart(self, top_n=15):
        """Create interactive bar chart for top jobs"""
        top_jobs = self.df['job'].value_counts().head(top_n)
        
        fig = px.bar(
            x=top_jobs.values,
            y=top_jobs.index,
            orientation='h',
            title=f'Top {top_n} Job Titles in India',
            labels={'x': 'Number of Postings', 'y': 'Job Title'},
            color=top_jobs.values,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            hovermode='closest',
            font=dict(size=12)
        )
        
        return fig.to_json()
    
    def create_location_map(self, top_n=20):
        """Create location distribution chart"""
        top_locations = self.df['location'].value_counts().head(top_n)
        
        fig = px.bar(
            x=top_locations.values,
            y=top_locations.index,
            orientation='h',
            title=f'Top {top_n} Job Locations',
            labels={'x': 'Number of Jobs', 'y': 'Location'},
            color=top_locations.values,
            color_continuous_scale='Plasma'
        )
        
        fig.update_layout(height=600, showlegend=False)
        
        return fig.to_json()
    
    def create_work_type_pie(self):
        """Create pie chart for work types"""
        work_types = self.df['work_type'].value_counts()
        
        fig = px.pie(
            values=work_types.values,
            names=work_types.index,
            title='Work Type Distribution',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        
        return fig.to_json()
    
    def create_company_followers_chart(self, top_n=15):
        """Create chart for companies by followers"""
        company_followers = self.df.groupby('company_name')['linkedin_followers'].mean().sort_values(ascending=False).head(top_n)
        
        fig = px.bar(
            x=company_followers.values,
            y=company_followers.index,
            orientation='h',
            title=f'Top {top_n} Companies by LinkedIn Followers',
            labels={'x': 'Average LinkedIn Followers', 'y': 'Company'},
            color=company_followers.values,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=600, showlegend=False)
        
        return fig.to_json()
    
    def create_applications_distribution(self):
        """Create histogram of applications"""
        fig = px.histogram(
            self.df,
            x='no_of_application',
            nbins=50,
            title='Distribution of Job Applications',
            labels={'no_of_application': 'Number of Applications'},
            color_discrete_sequence=['#636EFA']
        )
        
        fig.update_layout(height=400, showlegend=False)
        
        return fig.to_json()
    
    def create_posting_time_chart(self):
        """Create chart for job posting recency"""
        # Bin the posting times
        bins = [0, 24, 72, 168, 336, float('inf')]
        labels = ['< 1 day', '1-3 days', '3-7 days', '1-2 weeks', '> 2 weeks']
        
        self.df['posting_category'] = pd.cut(
            self.df['posted_hours_ago'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        posting_counts = self.df['posting_category'].value_counts().sort_index()
        
        fig = px.bar(
            x=posting_counts.index,
            y=posting_counts.values,
            title='Job Posting Recency',
            labels={'x': 'Time Period', 'y': 'Number of Jobs'},
            color=posting_counts.values,
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(height=400, showlegend=False)
        
        return fig.to_json()
    
    def create_top_companies_chart(self, top_n=15):
        """Create chart for top hiring companies"""
        top_companies = self.df['company_name'].value_counts().head(top_n)
        
        fig = px.bar(
            x=top_companies.values,
            y=top_companies.index,
            orientation='h',
            title=f'Top {top_n} Hiring Companies',
            labels={'x': 'Number of Job Postings', 'y': 'Company'},
            color=top_companies.values,
            color_continuous_scale='Greens'
        )
        
        fig.update_layout(height=600, showlegend=False)
        
        return fig.to_json()
    
    def create_dashboard_summary(self):
        """Create summary statistics for dashboard"""
        summary = {
            'total_jobs': int(len(self.df)),
            'total_companies': int(self.df['company_name'].nunique()),
            'total_locations': int(self.df['location'].nunique()),
            'avg_applications': float(self.df['no_of_application'].mean()),
            'remote_jobs': int(len(self.df[self.df['work_type'] == 'remote'])),
            'recent_jobs_24h': int(len(self.df[self.df['posted_hours_ago'] <= 24]))
        }
        
        return summary
    
    def create_all_visualizations(self):
        """Generate all visualizations with error handling"""
        print("📊 Creating all visualizations...")
        
        visualizations = {}
        
        try:
            visualizations['top_jobs'] = self.create_top_jobs_chart()
        except Exception as e:
            print(f"Error creating top jobs chart: {e}")
            visualizations['top_jobs'] = None
        
        try:
            visualizations['locations'] = self.create_location_map()
        except Exception as e:
            print(f"Error creating locations chart: {e}")
            visualizations['locations'] = None
        
        try:
            visualizations['work_types'] = self.create_work_type_pie()
        except Exception as e:
            print(f"Error creating work types chart: {e}")
            visualizations['work_types'] = None
        
        try:
            visualizations['company_followers'] = self.create_company_followers_chart()
        except Exception as e:
            print(f"Error creating company followers chart: {e}")
            visualizations['company_followers'] = None
        
        try:
            visualizations['applications'] = self.create_applications_distribution()
        except Exception as e:
            print(f"Error creating applications chart: {e}")
            visualizations['applications'] = None
        
        try:
            visualizations['posting_time'] = self.create_posting_time_chart()
        except Exception as e:
            print(f"Error creating posting time chart: {e}")
            visualizations['posting_time'] = None
        
        try:
            visualizations['top_companies'] = self.create_top_companies_chart()
        except Exception as e:
            print(f"Error creating top companies chart: {e}")
            visualizations['top_companies'] = None
        
        try:
            visualizations['summary'] = self.create_dashboard_summary()
        except Exception as e:
            print(f"Error creating summary: {e}")
            visualizations['summary'] = {}
        
        print("✓ All visualizations created")
        return visualizations

if __name__ == "__main__":
    viz = JobVisualizer()
    charts = viz.create_all_visualizations()
    print("\n✅ Visualization module ready!")