import requests
import json
import time

# Replace with your actual values
API_KEY = "opr9dyWzyka12j9DoBmw"  # Your PDFMonkey API Key
TEMPLATE_ID = "3EBFFE67-A105-4FB9-B75E-DD516DE4F11D"  # Your Document Template ID
JSON_FILE = r"C:\Users\udayv\OneDrive\Desktop\hackathon\fertilizer_recommendation (7).json"  # Your input JSON file
SAVE_PATH = r"C:\Users\udayv\OneDrive\Desktop\pdfs\output.pdf   "  # Where to save the PDF

def generate_pdf():
    """Generates a PDF document and returns its ID."""
    with open(JSON_FILE, 'r') as file:      
        data = json.load(file)

    url = "https://api.pdfmonkey.io/api/v1/documents"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"document": {"document_template_id": TEMPLATE_ID, "payload": data, "status": "pending"}}

    response = requests.post(url, headers=headers, json=payload)

    response_data = response.json()


    if response.status_code != 201: 
        print(f"❌ Error generating PDF: {response_data}")  
        return None

    document_id = response_data["document"]["id"]
    print(f"✅ PDF generation started. Document ID: {document_id}")
    return document_id

def wait_for_pdf(document_id):
    """Waits for the PDF to be generated and returns the download URL."""
    url = f"https://api.pdfmonkey.io/api/v1/documents/{document_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    for _ in range(12):  # Check every 5 seconds, up to 1 minute
        response = requests.get(url, headers=headers)
        response_data = response.json()

        if "document" in response_data:
            status = response_data["document"]["status"]
            download_url = response_data["document"]["download_url"]
            print(f"⏳ Status: {status}")

            if status == "success" and download_url:
                print(f"✅ PDF Ready! Downloading from: {download_url}")
                return download_url

        time.sleep(5)

    print("❌ PDF generation took too long or failed.")
    return None

def download_pdf(download_url):
    """Downloads the generated PDF from the given URL."""
    response = requests.get(download_url)  # ✅ No Authorization header here!

    if response.status_code == 200:
        with open(SAVE_PATH, "wb") as file:
            file.write(response.content)
        print(f"✅ PDF downloaded successfully: {SAVE_PATH}")
    else:
        print(f"❌ Error downloading PDF: {response.text}")

# Run the complete process
document_id = generate_pdf()
if document_id:
    download_url = wait_for_pdf(document_id)
    if download_url:
        download_pdf(download_url)
import requests
import json
import time

# Replace with your actual values
API_KEY = "opr9dyWzyka12j9DoBmw"  # Your PDFMonkey API Key
TEMPLATE_ID = "3EBFFE67-A105-4FB9-B75E-DD516DE4F11D"  # Your Document Template ID
JSON_FILE = r"C:\Users\udayv\OneDrive\Desktop\hackathon\fertilizer_recommendation (7).json"  # Your input JSON file
SAVE_PATH = r"C:\Users\udayv\OneDrive\Desktop\pdfs\output.pdf   "  # Where to save the PDF

def generate_pdf():
    """Generates a PDF document and returns its ID."""
    with open(JSON_FILE, 'r') as file:      
        data = json.load(file)

    url = "https://api.pdfmonkey.io/api/v1/documents"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"document": {"document_template_id": TEMPLATE_ID, "payload": data, "status": "pending"}}

    response = requests.post(url, headers=headers, json=payload)

    response_data = response.json()


    if response.status_code != 201: 
        print(f"❌ Error generating PDF: {response_data}")  
        return None

    document_id = response_data["document"]["id"]
    print(f"✅ PDF generation started. Document ID: {document_id}")
    return document_id

def wait_for_pdf(document_id):
    """Waits for the PDF to be generated and returns the download URL."""
    url = f"https://api.pdfmonkey.io/api/v1/documents/{document_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    for _ in range(12):  # Check every 5 seconds, up to 1 minute
        response = requests.get(url, headers=headers)
        response_data = response.json()

        if "document" in response_data:
            status = response_data["document"]["status"]
            download_url = response_data["document"]["download_url"]
            print(f"⏳ Status: {status}")

            if status == "success" and download_url:
                print(f"✅ PDF Ready! Downloading from: {download_url}")
                return download_url

        time.sleep(5)

    print("❌ PDF generation took too long or failed.")
    return None

def download_pdf(download_url):
    """Downloads the generated PDF from the given URL."""
    response = requests.get(download_url)  # ✅ No Authorization header here!

    if response.status_code == 200:
        with open(SAVE_PATH, "wb") as file:
            file.write(response.content)
        print(f"✅ PDF downloaded successfully: {SAVE_PATH}")
    else:
        print(f"❌ Error downloading PDF: {response.text}")

# Run the complete process
document_id = generate_pdf()
if document_id:
    download_url = wait_for_pdf(document_id)
    if download_url:
        download_pdf(download_url)





by Krish Arora
