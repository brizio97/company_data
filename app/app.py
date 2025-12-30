from utils import company_name_from_number, company_incorporation_date_from_number, create_tree_graph, company_search
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import os
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

class CompanySearch(FlaskForm):
 company_searched = StringField('Search company name or number', validators=[DataRequired()])
 submit = SubmitField()


from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)
app.config['SECRET_KEY'] = 'password' # for the form to work

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content

@app.route('/', methods = ["GET", "POST"])
def index():
    form = CompanySearch()
    company_searched = None
    
    if form.validate_on_submit():
        company_searched = form.company_searched.data
        return redirect(url_for('search', company_searched=company_searched))

    return render_template('index.html', form=form, company_searched = company_searched)


@app.route('/search/<company_searched>', methods = ["GET", "POST"])
def search(company_searched):
    items = company_search(company_searched)
    return render_template('company_searched.html', company_searched=company_searched, items=items)


@app.route('/company/<company_number>')
def shareholders(company_number):
 selected_date = request.args.get('selected_date', None)
 # Convert empty string to None
 if selected_date == '':
     selected_date = None
     
 # Get incorporation date
 incorporation_date = company_incorporation_date_from_number(company_number)
     
 max_level_param = request.args.get('max_level', '2')
 
 # Convert max_level: 'all' -> 999, otherwise convert to int
 if max_level_param == 'all':
     max_level = 999
 else:
     try:
         max_level = int(max_level_param)
     except ValueError:
         max_level = 2
 
 shareholders_tree = create_tree_graph(company_number, selected_date=selected_date, max_level=max_level)
 print('tree created')
 return render_template('company.html', shareholders_tree = shareholders_tree, company_name = company_name_from_number(company_number), company_number = company_number, selected_date = selected_date, max_level = max_level_param, incorporation_date = incorporation_date)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)  #prod 8080
