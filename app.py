import os, sqlite3
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from gensim.models import Word2Vec
import joblib
from utils import genVec, tokenizeTxt
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///jobs.db'
db = SQLAlchemy(app)

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    webIndex = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(50), nullable=False)
    

@app.route('/')
def index():
    label_filter = request.args.get('label')
    if label_filter:
        jobs = Job.query.filter_by(label=label_filter).all()
    else:
        jobs = Job.query.all()
    categories = ['Engineering', 'Healthcare_Nursing', 'Accounting_Finance', 'Sales']
    return render_template('index.html', jobs=jobs, categories=categories, selected_label=label_filter)

@app.route('/job/<int:job_id>')
def job_detail(job_id):
    job = Job.query.get_or_404(job_id)
    return render_template('job_detail.html', job=job)

@app.route('/create', methods=['GET', 'POST'])
def create_job():
    categories = ['Engineering', 'Healthcare_Nursing', 'Accounting_Finance', 'Sales']
    if request.method == 'POST':
        # Classify the content
        if request.form['button'] == 'Classify':
            # Read the content
            f_title = request.form['title']
            f_content = request.form['description']
            f_company = request.form['company']
            f_category = request.form['category']
            
            # Tokenize the content of the .txt file so as to input to the saved mode
            tokenized_data = tokenizeTxt(f_content)
            
            # Load the LR model
            model = joblib.load("models/LR_description_model.pkl")
            
            vocab_des = {}
            with open('vocab.txt', 'r') as file:
                for line in file:
                    word, idx = line.strip().split(':')
                    vocab_des[word] = int(idx)
            tfidf_vec_des = TfidfVectorizer(vocabulary=vocab_des)
            
            # Fit and transform descriptions using the TfidfVectorizer
            X_tfidf_des = tfidf_vec_des.fit_transform(tokenized_data)

            # Predict the label of tokenized_data
            y_pred = model.predict(X_tfidf_des)
            y_pred = y_pred[0]
            print('category', f_category)
            return render_template('create_job.html', prediction=y_pred, title=f_title, description=f_content, categories=categories, company=f_company, selectedCategory=f_category)
        elif request.form['button'] == 'Save':
            title = request.form['title']
            description = request.form['description']
            category = request.form['category']
            company = request.form['company']
            
            new_job = Job(title=title, description=description, category=category, company=company)
            db.session.add(new_job)
            db.session.commit()
        
        return redirect(url_for('index'))
    return render_template('create_job.html', categories=categories, selectedCategory="Engineering")


@app.route('/search', methods = ['POST'])
def search():
    if request.method == 'POST': # If user search
        folder_path = 'instance'
        db_path = os.path.join(folder_path, 'jobs.db')
        conn = sqlite3.connect(db_path) # Get the data from jobs.db
        conn.row_factory = sqlite3.Row # Make sure it in row format
        cursor = conn.cursor()
        if request.form['search'] == '': # If the user click search
            search_string = request.form["searchword"] # Get the string from search
            # Execute the SQL query to get the data likely with the searching string
            cursor.execute('''
                           SELECT * 
                           FROM job
                           WHERE title LIKE ? OR description LIKE ? OR company LIKE ? OR label LIKE ?
                            ''', (f'%{search_string}%', f'%{search_string}%', f'%{search_string}%', f'%{search_string}%'))
            # Get all the jobs from the execute SQL query
            jobs = cursor.fetchall()
            conn.close()

            return render_template('search.html', jobs = jobs) # Get to the search.html with all the jobs similar with searching string
   
    else: # Else go to home page
        return render_template('home.html')
    
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)