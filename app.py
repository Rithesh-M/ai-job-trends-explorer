"""
Flask Web Application for AI-Driven Job Trends Explorer
Main application file
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
import pandas as pd
import sys
import os
from functools import wraps
from datetime import timedelta

# Add parent directory to path to access dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_analysis import JobDataAnalyzer
from ai_recommender import JobRecommender
from visualizations import JobVisualizer

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production-12345'  # Change this in production!
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
CORS(app)

# Simple user database (in production, use a real database)
USERS = {
    'admin': 'admin123',
    'user': 'user123',
    'demo': 'demo123'
}

# Initialize components
print("🚀 Initializing AI Job Trends Explorer...")

# Global variables for models
recommender = None
visualizer = None
analyzer = None

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please login to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def initialize_app():
    """Initialize all components"""
    global recommender, visualizer, analyzer
    
    try:
        # Get the correct path to the CSV file
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned_linkedin_jobs.csv')
        
        # Check if CSV exists
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found at {csv_path}")
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        
        print(f"Using dataset: {csv_path}")
        
        # Initialize visualizer
        visualizer = JobVisualizer(csv_path=csv_path)
        
        # Initialize analyzer
        analyzer = JobDataAnalyzer(csv_path=csv_path)
        
        # Initialize recommender
        recommender = JobRecommender(csv_path=csv_path)
        
        # Try to load existing model, otherwise train new one
        if not recommender.load_model():
            print("Training new AI model...")
            recommender.train_model()
        
        print("✅ Application initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing app: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.route('/')
def index():
    """Home page - redirect to login if not authenticated"""
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            session.permanent = True
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard page with visualizations"""
    try:
        # Get summary statistics
        summary = visualizer.create_dashboard_summary()
        return render_template('dashboard.html', summary=summary, username=session.get('username'))
    except Exception as e:
        print(f"Dashboard error: {e}")
        # Return with empty summary instead of showing error message
        summary = {
            'total_jobs': 0,
            'total_companies': 0,
            'total_locations': 0,
            'avg_applications': 0,
            'remote_jobs': 0,
            'recent_jobs_24h': 0
        }
        return render_template('dashboard.html', summary=summary, username=session.get('username'))

@app.route('/search')
@login_required
def search_page():
    """Search page"""
    return render_template('search.html', username=session.get('username'))

@app.route('/recommendations')
@login_required
def recommendations_page():
    """Recommendations page"""
    return render_template('recommendations.html', username=session.get('username'))

@app.route('/api/visualizations')
@login_required
def get_visualizations():
    """API endpoint for all visualizations"""
    try:
        charts = visualizer.create_all_visualizations()
        return jsonify(charts)
    except Exception as e:
        print(f"Visualization error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
@login_required
def search_jobs():
    """API endpoint for job search"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        location = data.get('location', 'all')
        work_type = data.get('work_type', 'all')
        limit = int(data.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Get recommendations
        results = recommender.get_recommendations(
            query=query,
            top_n=limit,
            location_filter=location if location != 'all' else None,
            work_type_filter=work_type if work_type != 'all' else None
        )
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
@login_required
def get_stats():
    """API endpoint for statistics"""
    try:
        stats = analyzer.get_summary_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/top-skills')
@login_required
def get_top_skills():
    """API endpoint for top skills"""
    try:
        top_n = int(request.args.get('limit', 30))
        skills = recommender.extract_top_skills(top_n=top_n)
        
        # Format for frontend
        skills_data = [{'skill': skill, 'score': float(score)} for skill, score in skills]
        
        return jsonify({
            'success': True,
            'skills': skills_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/filters')
@login_required
def get_filters():
    """API endpoint for filter options"""
    try:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned_linkedin_jobs.csv')
        df = pd.read_csv(csv_path)
        
        # Get unique locations (top 50)
        locations = df['location'].value_counts().head(50).index.tolist()
        
        # Get work types
        work_types = df['work_type'].unique().tolist()
        
        return jsonify({
            'locations': locations,
            'work_types': work_types
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """404 error handler"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    """500 error handler"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    initialize_app()
    print("\n🌐 Starting Flask server...")
    print("📍 Access the app at: http://localhost:5001")
    print("\n")
    app.run(debug=True, host='0.0.0.0', port=5001)