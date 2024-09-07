from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

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
    jobs = Job.query.all()
    return render_template('index.html', jobs=jobs)

@app.route('/job/<int:job_id>')
def job_detail(job_id):
    job = Job.query.get_or_404(job_id)
    return render_template('job_detail.html', job=job)

@app.route('/category/<string:category>')
def category(category):
    jobs = Job.query.filter_by(category=category).all()
    return render_template('categories.html', jobs=jobs, category=category)

@app.route('/create', methods=['GET', 'POST'])
def create_job():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        category = request.form['category']
        salary = request.form['salary']
        
        new_job = Job(title=title, description=description, category=category, salary=salary)
        db.session.add(new_job)
        db.session.commit()
        
        return redirect(url_for('index'))
    return render_template('create_job.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)