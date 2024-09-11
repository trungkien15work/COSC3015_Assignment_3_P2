from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from gensim.models import Word2Vec
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jobs.db'
db = SQLAlchemy(app)

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    salary = db.Column(db.String(50), nullable=True)

@app.route('/')
def index():
    category_filter = request.args.get('category')
    if category_filter:
        jobs = Job.query.filter_by(category=category_filter).all()
    else:
        jobs = Job.query.all()
    categories = ['Engineering', 'Healthcare Nursing', 'Accounting Finance', 'Sales']
    return render_template('index.html', jobs=jobs, categories=categories, selected_category=category_filter)

@app.route('/job/<int:job_id>')
def job_detail(job_id):
    job = Job.query.get_or_404(job_id)
    return render_template('job_detail.html', job=job)

@app.route('/create', methods=['GET', 'POST'])
def create_job():
    categories = ['Engineering', 'Healthcare Nursing', 'Accounting Finance', 'Sales']
    if request.method == 'POST':
        # Classify the content
        if request.form['button'] == 'Classify':
            # Read the content
            f_title = request.form['title']
            f_content = request.form['description']
            f_salary = request.form['salary']
            f_category = request.form['category']
            
            # Load the LR model
            model = joblib.load("models/LR_description_model.pkl")
            
            # Load the tfidf vecto
            tfidf_vec_des = joblib.load('models/description_tfidf.joblib')
            
            # Fit and transform descriptions using the TfidfVectorizer
            X_tfidf_des = tfidf_vec_des.fit_transform([f_content])

            # Predict the label of tokenized_data
            y_pred = model.predict(X_tfidf_des)
            y_pred = y_pred[0]
            
            return render_template('create_job.html', prediction=y_pred, title=f_title, description=f_content, categories=categories, salary=f_salary, selectedCategory=f_category)
        elif request.form['button'] == 'Save':
            title = request.form['title']
            description = request.form['description']
            category = request.form['category']
            salary = request.form['salary']
            
            new_job = Job(title=title, description=description, category=category, salary=salary)
            db.session.add(new_job)
            db.session.commit()
        
        return redirect(url_for('index'))
    return render_template('create_job.html', categories=categories, selectedCategory="Engineering")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)