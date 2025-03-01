from utils import confirmation_statement_to_data, does_company_number_exist
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


class CompanyNumber(FlaskForm):
 company_number_form_input = StringField('Enter Company Number', validators=[DataRequired()])
 submit = SubmitField('Search')


from flask import Flask, render_template, redirect, url_for
app = Flask(__name__)
app.config['SECRET_KEY'] = 'password'
bootstrap = Bootstrap(app)

@app.route('/', methods = ["GET", "POST"])
def index():
 form = CompanyNumber()
 company_number_form_input = None
 does_company_exist = None
 
 if form.validate_on_submit():
     company_number_form_input = form.company_number_form_input.data
     if does_company_number_exist(company_number_form_input) == '1':
         return redirect(url_for('shareholders', company_number=company_number_form_input))
     else:
         does_company_exist = '0'
      
 return render_template('index.html', form=form, company_number_form_input=company_number_form_input, does_company_exist = does_company_exist)

@app.route('/<company_number>')
def shareholders(company_number):
 return render_template('company.html', shareholders_table = confirmation_statement_to_data(company_number))

if __name__ == '__main__':
    app.run(host='localhost', port=5001, debug=True)