import pytesseract
import requests
import json
import pandas as pd
import re
from PIL import Image
import io
import fitz
import PyPDF2
import google.generativeai as genai
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'


# Set up gemini LLM 
gemini_api_key = 'AIzaSyACKA-uC5lsOA2zJ1__XdfdAQmbeoOHkjA'
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel(model_name = 'gemini-1.5-flash')
generation_config = {
  "temperature": 0, "response_mime_type": "application/json"}


# Key for Companies house API
key = '5c9a7f45-2045-4c0c-8b50-a2a3268bd8ff'


# Gets list of documents per company number for a given company

def filing_list_per_company_number(company_number):
    results = 100
    result_counter=-100
    filelist = []
    while results > 99:
        result_counter = result_counter + 100
        params = {
        'items_per_page' : '101',
        'start_index' : '{}'.format(result_counter)
        }
        r = requests.get('https://api.company-information.service.gov.uk/company/'+company_number+'/filing-history', auth=(key, ''), params=params)
        filinghistory = r.json()
        items = filinghistory.get('items')
        filelist = filelist + items
        results = len(items)
    return filelist


def does_company_number_exist(company_number):
        if len(filing_list_per_company_number(company_number)) == 0:
            return '0'
        else:
            return '1'
        
def company_name_from_number(company_number):
    r = requests.get('https://api.company-information.service.gov.uk/company/'+company_number, auth=(key, ''))
    company_information = r.json()
    company_name = company_information.get('company_name')
    return(company_name)
 
# Creates a pandas dataframe for the shareholders of a company, based on the confirmation statement

def confirmation_statement_to_data(company_number):
    filelist = filing_list_per_company_number(company_number)
    filtered = [d for d in filelist if d['description'] == 'confirmation-statement-with-updates']
    output_columns = ["Number of Shares", "Type of Shares", "Name"]
    output_df = pd.DataFrame(columns=output_columns)
    matches = ""

    for n in range(len(filtered)):
        confirmation = filtered[n]['links']['document_metadata']
        r = requests.get(confirmation+'/content', auth=(key, ''))
        confirmation_pdf = r.content
    # extract text
        pdf_file = io.BytesIO(confirmation_pdf)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        page = pdf_reader.pages[1]
        if page.extract_text() == '': # If no text, use OCR or GenAI
            has_pdf_text = 0
        elif page.extract_text() != '': #If text available, use pdf text extract logic
            has_pdf_text = 1

        extracted_text = ''

        if has_pdf_text == 1:
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text = extracted_text + '///' + page.extract_text()
        elif has_pdf_text == 0:
            pdf_document = fitz.open(stream=confirmation_pdf, filetype="pdf")
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(dpi=300)  # Adjust DPI for better quality
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img, config = r'--psm 6')
                extracted_text = extracted_text + f"Text from page {page_num + 1}:\n{text}\n"
                if "Electronically filed document" not in extracted_text: # if not electronically filed document, skip and go to Gemini
                    break

        # Extract data from text using regex
        pattern = (
            r"Shareholding \d+: (\d+) (.+?) shares held as at the date of.*?\n"
            r".*?Name: (.+?)(?=\nShareholding|\n)"
        )
        matches = re.findall(pattern, extracted_text, re.DOTALL)
        #if no matches, use Gemini
        if len(matches) == 0:
            data = pd.DataFrame(columns=output_columns)
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(dpi=300)  # Adjust DPI for better quality
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)    
                prompt_parts = [img, "Based on the image, create a table, in json output, with the following columns: Number of Shares, Type of Shares, Name. Let Name be the full name of the shareholder. Let the main dictionary be named 'shareholders'."]
                genai_response = model.generate_content(prompt_parts, generation_config = generation_config)
                genai_response_json = json.loads(genai_response.text)
                data_genai = pd.DataFrame(genai_response_json['shareholders'])
                data = pd.concat([data, data_genai], ignore_index=True, axis=0)
        elif len(matches) > 0:
            data = pd.DataFrame(matches, columns = output_columns)

    # Add document name and description
        data['Document Date'] = filtered[n]['date']
        data['Document Name'] = filtered[n]['description']
        output_df = pd.concat([output_df, data], ignore_index=True, axis=0)

        # Display the DataFrame
    output_df = output_df[output_df["Number of Shares"].notnull() & output_df["Type of Shares"].notnull() & output_df["Name"].notnull()]

    return(output_df.to_html(classes="table table-striped", index=False))








