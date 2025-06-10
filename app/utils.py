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
#pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
import time
import datetime
from thefuzz import fuzz
import logging
from requests.exceptions import SSLError
from pyvis.network import Network
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import networkx as nx
from functools import lru_cache

# Set up gemini LLM 
gemini_api_key = 'AIzaSyACKA-uC5lsOA2zJ1__XdfdAQmbeoOHkjA'
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel(model_name = 'gemini-1.5-flash')
generation_config = {
  "temperature": 0, "response_mime_type": "application/json"}


# Key for Companies house API
key = '5c9a7f45-2045-4c0c-8b50-a2a3268bd8ff'



logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True) # force not needed, just to change this without restarting the kernel




def requests_get(url, params=None, auth = (key, ''), retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, auth=auth, timeout=10)
            return response
        except Exception as e:
            logging.warning(f"Error: {e}")
            if attempt < retries - 1:
                logging.warning(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.critical("Final attempt failed.")
                raise




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
        r = requests_get('https://api.company-information.service.gov.uk/company/'+company_number+'/filing-history', params=params)
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

@lru_cache(maxsize=128)
def company_name_from_number(company_number):
    r = requests_get('https://api.company-information.service.gov.uk/company/' + company_number)
    r = r.json()
    return(r.get('company_name'))
 


def extract_images_from_pdf(pdf_document):
    page_images = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=150)  # Adjust DPI for better quality
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_images.append(img)
    return page_images

def process_image_llm(img, prompt):
    prompt_parts = [img, prompt]
    genai_response = model.generate_content(prompt_parts, generation_config = generation_config)
    genai_response_json = json.loads(genai_response.text)
    data_genai = pd.DataFrame(genai_response_json['shareholders'])
    return(data_genai)
  

def ocr_from_pdf_image(img):
    text = pytesseract.image_to_string(img, config = r'--psm 6')
    if "Electronically filed document" not in text: # if not electronically filed document, skip and go to Gemini
        logging.debug('Document not electronically filed therefore not standard format. Go to Gemini to extract information.')
        return 'FAIL'
    return text


# Creates a pandas dataframe for the shareholders of a company, based on the confirmation statement


def confirmation_statement_to_data(company_number):
    start = time.time()
    logging.debug('Begin confirmation_statement_to_data('+ company_number + ')')
    filelist = filing_list_per_company_number(company_number)
    filtered = [d for d in filelist if d['description'] == 'confirmation-statement-with-updates']
    output_columns = ["Number of Shares", "Type of Shares", "Name"]
    output_df = pd.DataFrame(columns=output_columns)
    matches = ""
    prompt = "Based on the image, create a table, in json output, with the following columns: Number of Shares, Type of Shares, Name. Let Name be the full name of the shareholder. If the page says 'statement of capital' then return an empty table. Let the main dictionary be named 'shareholders'."
    document_count = range(len(filtered))

    for n in document_count:
        logging.debug('Begin document ' + str(n+1) + ' of ' + str(max(document_count)+1))
        confirmation = filtered[n]['links']['document_metadata']
        r = requests_get(confirmation+'/content')
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
            logging.debug('Text present in pdf. Proceed to extract text with pdf reader.')
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text = extracted_text + '///' + page.extract_text()
        elif has_pdf_text == 0:
            logging.debug('No text present in pdf. Proceed to extract images from pdf file.')
            pdf_document = fitz.open(stream=confirmation_pdf, filetype="pdf")
            page_images = extract_images_from_pdf(pdf_document)

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(ocr_from_pdf_image, img) for img in page_images]
                for future in as_completed(futures):
                        text = future.result()
                        if text == 'FAIL': # if not electronically filed document
                            logging.debug('Document not electronically filed therefore not standard format. Go to Gemini to extract information.')
                            #break
                        extracted_text = extracted_text + f"\n{text}\n"

        # Extract data from text using regex
        pattern = (
            r"Shareholding \d+: (\d+) (.+?) shares held as at the date of.*?\n"
            r".*?Name: (.+?)(?=\nShareholding|\n)"
        )
        matches = re.findall(pattern, extracted_text, re.DOTALL)
        #if no matches, use Gemini
        if len(matches) == 0:
            logging.debug('No regex matches. Go to Gemini to extract information from images.')
            data = pd.DataFrame(columns=output_columns)
            pdf_document = fitz.open(stream=confirmation_pdf, filetype="pdf")
            page_images = extract_images_from_pdf(pdf_document)
            data_frames = []
            with ThreadPoolExecutor(max_workers=20) as executor: 
                futures = [executor.submit(process_image_llm, img, prompt) for img in page_images]
                for future in as_completed(futures):
                        df = future.result()
                        data_frames.append(df)
            data = pd.concat(data_frames, ignore_index=True)
        elif len(matches) > 0:
            logging.debug('Regex matches found.')
            data = pd.DataFrame(matches, columns = output_columns)

    # Add document name and description
        data['Document Date'] = pd.to_datetime(filtered[n]['date'])
        data['Document Name'] = filtered[n]['description']
        data['Document ID'] = filtered[n]['transaction_id']
        output_df = pd.concat([output_df, data], ignore_index=True, axis=0)
        logging.debug('Completed document ' + str(n+1) + ' of ' + str(max(document_count)+1))
        # Display the DataFrame
    output_df = output_df[output_df["Number of Shares"].notnull() & output_df["Type of Shares"].notnull() & output_df["Name"].notnull()]
    output_df = output_df[~output_df['Type of Shares'].str.contains("transfer", case=False)]
    output_df['Company Number'] = company_number
    output_df['Company Name'] = company_name_from_number(company_number)
    end = time.time()
    logging.debug('End confirmation_statement_to_data('+ company_number + f'), after {end - start:.2f} seconds')
    return(output_df)

   



def incorporation_to_data(company_number):
    start = time.time()
    logging.debug('Begin incorporation_to_data(' + company_number + ')')
    filelist = filing_list_per_company_number(company_number)
    filtered = [d for d in filelist if d['description'] == 'incorporation-company']
    output_columns = ["Number of Shares", "Type of Shares", "Name"]
    output_df = pd.DataFrame(columns=output_columns)
    prompt = "Based on the image, create a table, in json output, with the following columns with the exact names: Number of Shares, Type of Shares, Name. Let Name be the full name of the shareholder. Let the main dictionary be named 'shareholders'. You are looking for the initial shareholders of the company"
    document_count = range(len(filtered))
    for n in document_count:
        logging.debug('Begin document ' + str(n+1) + ' of ' + str(max(document_count)+1))
        incorporation = filtered[n]['links']['document_metadata']
        r = requests_get(incorporation+'/content')
        incorporation_pdf = r.content
    # extract text
        pdf_file = io.BytesIO(incorporation_pdf)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        data = pd.DataFrame(columns=output_columns)
        pdf_document = fitz.open(stream=incorporation_pdf, filetype="pdf")
        page_images = extract_images_from_pdf(pdf_document)
        
        data_frames = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(process_image_llm, img, prompt) for img in page_images]
            for future in as_completed(futures):
                    df = future.result()
                    data_frames.append(df)
        data = pd.concat(data_frames, ignore_index=True)

    # Add document name and description
        data['Document Date'] = pd.to_datetime(filtered[n]['date'])
        data['Document Name'] = filtered[n]['description']
        data['Document ID'] = filtered[n]['transaction_id']
        output_df = pd.concat([output_df, data], ignore_index=True, axis=0)
        logging.debug('Completed document ' + str(n+1) + ' of ' + str(max(document_count)+1))
        # Display the DataFrame
    output_df = output_df[output_df["Number of Shares"].notnull() & output_df["Type of Shares"].notnull() & output_df["Name"].notnull()]
    output_df['Company Number'] = company_number
    output_df['Company Name'] = company_name_from_number(company_number)
    end = time.time()
    logging.debug('End incorporation_to_data(' + company_number +  f'), after {end - start:.2f} seconds')
    return(output_df)



def company_shareholding(company_number):
    start = time.time()
    logging.debug('Begin company_shareholding(' + company_number + ')')
    incorporation_data = incorporation_to_data(company_number)
    confirmation_statement_data = confirmation_statement_to_data(company_number)
    company_shareholding = pd.concat([incorporation_data, confirmation_statement_data], ignore_index = True, axis = 0)
    if len(company_shareholding) == 0:
        end = time.time()
        logging.debug('End company_shareholding(' + company_number + f'), after {end - start:.2f} seconds. No shareholders found.')
        return (company_shareholding)

    company_shareholding['Number of Shares'] = pd.to_numeric(company_shareholding['Number of Shares'], errors = 'coerce')
    share_sums = company_shareholding.groupby(['Document ID']).agg(total_shares = ('Number of Shares', 'sum'))
    company_shareholding = pd.merge(company_shareholding, share_sums, how = 'left', on = 'Document ID')
    company_shareholding['Percentage of Total Shares'] = company_shareholding['Number of Shares'] / company_shareholding['total_shares']
    company_shareholding = company_shareholding.sort_values(by = 'Document Date')
    documents = company_shareholding[['Document ID', 'Document Date']].drop_duplicates().reset_index(drop=True)
    documents['Document Valid To Date'] = documents['Document Date'].shift(-1).fillna(datetime.datetime(2099,1,1))
    documents = company_shareholding[['Document ID', 'Document Date']].drop_duplicates().reset_index(drop=True)
    documents.rename(columns={'Document Date': 'Document Date 2'}, inplace=True)
    documents['Document Valid To Date'] = documents['Document Date 2'].shift(-1).fillna(datetime.datetime(2099,1,1))
    company_shareholding_enhanced = (pd.merge(company_shareholding, documents, how = 'left', on = 'Document ID'))[
        [
        'Company Number',
        'Company Name',
        'Name',
        'Number of Shares',
        'Percentage of Total Shares',
        'Type of Shares',
        'Document Date',
        'Document Valid To Date',
        'Document Name',
        'Document ID'
        ]]
    end = time.time()
    logging.debug('End company_shareholding(' + company_number + f'), after {end - start:.2f} seconds')
    return (company_shareholding_enhanced)




def company_number_from_name_search(searched_name):
    start = time.time()
    logging.debug('Begin company_number_from_name_search(' + searched_name + ')')
    r = requests_get('https://api.company-information.service.gov.uk/search', params={'items_per_page' : '10', 'q' : searched_name})
    content = r.json()
    items = content.get('items')
    rank = 0
    output_columns = ['rank', 'company_number', 'name']
    result_list_df = pd.DataFrame(columns=output_columns)

    for i in items:
        rank = rank + 1
        company_number = i.get('company_number')
        if company_number is None:
            continue
        name = i.get('title')
        previous_name = i.get('snippet')
        data = pd.DataFrame([{
            'rank': rank,
            'company_number': company_number,
            'name': name,
            'previous_name': previous_name
        }])
        result_list_df = pd.concat([result_list_df, data], ignore_index=True, axis=0)
    logging.debug('Raw data extracted:')
    logging.debug(result_list_df)
    if len(result_list_df) == 0:
        logging.debug('No Match')
        return('No Match')
    #Remove whitespace and capitalisation then do fuzzy comparison
    result_list_df['searched_company_name'] = searched_name.replace(' ','').lower()
    result_list_df['name'] = result_list_df['name'].str.lower().str.replace(' ','', regex=False).str.replace('ltd','limited', regex=False)
    result_list_df['previous_name'] = result_list_df['previous_name'].str.lower().str.replace(' ','', regex=False).str.replace('ltd','limited', regex=False)
    #Calculate fuzz ratio twice. Once between company names and searched name, and the second time using the previous company name.
    result_list_df['fuzz_ratio_1'] = result_list_df.apply(lambda x: fuzz.ratio(x['searched_company_name'], x['name']), axis=1)
    result_list_df['fuzz_ratio_2'] = result_list_df.apply(lambda x: fuzz.ratio(x['searched_company_name'], x['previous_name']), axis=1)
    result_list_df['fuzz_ratio'] = result_list_df[['fuzz_ratio_1', 'fuzz_ratio_2']].max(axis=1)

    #Only show the cases where fuzz ration is 100. This is only a perfect match.
    result_list_df = result_list_df[result_list_df['fuzz_ratio'] == 100].reset_index(drop=True)
    end = time.time()
    logging.debug('End company_number_from_name_search(' + searched_name + '). Final outcome:')
    logging.debug(result_list_df)
    if len(result_list_df) == 1:
        logging.debug(result_list_df.at[0, 'company_number'])
        return(result_list_df.at[0, 'company_number'])  # return a dictionary containing previous and current name.
    else:
        logging.debug('No Match')
        return('No Match')




def full_shareholder_tree(company_number, max_level, visited=None, level=0):
    logging.debug('level ' + str(level))
    if visited is None:
        visited = set()
    if company_number in visited:
        return pd.DataFrame()  # Already processed
    visited.add(company_number)
    logging.debug('Visited:')
    logging.debug(visited)
    shareholders_output = company_shareholding(company_number)
    shareholders_output["Hierarchy Level"] = level
    shareholder_list = list(set(shareholders_output['Name'].tolist()))
    if level == max_level:
        return (shareholders_output) #Reached max level
    logging.debug(shareholder_list)
    company_numbers_found = []

    # Take the shareholder list and search them in companies house to get their number, using multiple threads.
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(company_number_from_name_search, i) for i in shareholder_list]

    for future in as_completed(futures):
        result = future.result()
        company_numbers_found.append(result)

    # Once we have all the numbers, iterate through them and recursively call full_shareholder_tree.
    for i in company_numbers_found:
        if i == 'No Match':
            continue
        shareholders = full_shareholder_tree(i, max_level, visited, level+1)
        shareholders_output = pd.concat([shareholders_output, shareholders]).drop_duplicates().reset_index(drop=True)
    return(shareholders_output)


def create_tree_graph(company_number):
    start = time.time()
    logging.debug('Begin create tree graph (' + str(company_number) + ')')
    tree = full_shareholder_tree(company_number, 2)
    tree_filtered = tree[(tree['Document Date'] <= datetime.datetime(2025,1,1)) & (tree['Document Valid To Date'] > datetime.datetime(2025,1,1))]

    n = nx.MultiDiGraph()

    tree_leaf = tree_filtered[['Name', 'Hierarchy Level']]
    tree_branch = tree_filtered[['Company Name', 'Hierarchy Level']].rename(columns={"Company Name": "Name"})
    tree_branch['Hierarchy Level'] = tree_branch['Hierarchy Level'] - 1

    tree_agg = pd.concat([tree_branch, tree_leaf], axis=0)

    all_nodes = tree_agg.groupby('Name').agg({'Hierarchy Level': 'max'})

    for node in all_nodes.index:
        n.add_node(node, label=node, shape = 'dot', level=int(all_nodes.loc[node, 'Hierarchy Level']))

    for _, row in tree_filtered.iterrows():
        n.add_edge(row['Name'], 
                    row['Company Name'], 
                    title=row['Type of Shares'] + ' ' + str(round(row['Percentage of Total Shares'] * 100, 3)) + '%', 
                    value = row['Percentage of Total Shares']
                    )

    net = Network(directed=True, height='600px', width='100%', bgcolor='#FFFFFF', font_color='black')
    net.from_nx(n)

    net.set_edge_smooth('dynamic')

    net.set_options("""
    {
        "layout": {
        "hierarchical": {
            "enabled": false,
            "direction": "DU",
            "sortMethod": "directed",
            "nodeSpacing": 200,
            "treeSpacing": 500,
            "levelSeparation":500
        }
        },
        "nodes": {
        "font": {
            "size": 8
        }
        },
        "edges": {
        "arrows": {
            "to": { "enabled": true },
            "scaleFactor": 1
        },
        "arrowStrikethrough": "false"
        },
        "physics": {
        "enabled": true,
        "solver": "repulsion",
        "repulsion": {
            "nodeDistance": 80,
            "centralGravity": 0.0,
            "springLength": 150,
            "springConstant": 0.01,
            "damping": 0.1
        }
        }             
    }
    """)
    end = time.time()
    logging.debug('End create tree graph (' + str(company_number) + f'), after {end - start:.2f} seconds')
    #net.save_graph('shareholder_network.html')
    return (net.generate_html())