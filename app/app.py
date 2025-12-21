from utils import confirmation_statement_to_data, does_company_number_exist, company_name_from_number, create_tree_graph, company_search
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import os


class CompanySearch(FlaskForm):
 company_searched = StringField('Search company name or number', validators=[DataRequired()])
 submit = SubmitField()



from flask import Flask, render_template, redirect, url_for
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
 shareholders_tree = create_tree_graph(company_number)
 print('tree created')
 return render_template('company.html', shareholders_tree = shareholders_tree, company_name = company_name_from_number(company_number), company_number = company_number)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)  #prod 8080
