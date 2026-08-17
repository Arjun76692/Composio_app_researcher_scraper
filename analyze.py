import pandas as pd
import os
import time
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq Client
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

def analyze_and_generate_html():
    print("Starting Analysis and Report Generation...")
    
    # 1. Load Data
    try:
        df = pd.read_excel("research_results_checkpoint.xlsx")
        print(f"Loaded {len(df)} records from research_results_checkpoint.xlsx")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    # Load accuracy sample
    accuracy_data = ""
    try:
        vdf = pd.read_excel("verification_sample.xlsx")
        cols = [c for c in vdf.columns if c.startswith('manual_') and c.endswith('_correct')] + ['manual_evidence_valid']
        stats = []
        for c in cols:
            yes = (vdf[c].astype(str).str.strip().str.lower() == 'yes').sum()
            total = len(vdf[vdf[c].notna() & (vdf[c].astype(str).str.strip() != '')])
            if total > 0:
                stats.append(f"{c}: {yes}/{total} ({(yes/total)*100:.1f}%)")
        
        accuracy_data = "Accuracy Manual Verification Results:\n" + "\n".join(stats)
        print(accuracy_data)
    except Exception as e:
        print(f"Error loading verification sample: {e}")

    # Convert dataframe to a JSON string for the LLM
    data_json = df.to_json(orient='records')
    
    # 2. Prepare Prompt
    prompt = f"""
    You are a Product Ops Analyst at Composio.
    I have provided you with research data for multiple applications, as well as the manual accuracy verification results.
    
    Your task is to write a comprehensive HTML report summarizing the research and the accuracy improvements.
    
    The HTML report MUST include:
    1. A professional title and header.
    2. An executive summary of the patterns found (common auth methods, self-serve vs gated, buildability).
    3. An Accuracy Summary Section structured exactly as follows:
       - Sample size and method ("We manually verified X apps...")
       - Headline accuracy number (Extract from the Accuracy Results below)
       - What broke and why (Discuss earlier failures: MCP-feature bias on Salesforce, unofficial GitHub repos, empty rows)
       - What you fixed and the resulting delta (Discuss schema validation, domain filtering, empty-row guards)
       - Remaining known limitations (Community repos still creep in, LLM confidence isn't perfectly calibrated)
    4. A beautiful, modern, clean UI with inline CSS styling (no Tailwind CSS, just vanilla CSS). Use a sleek, premium design.
    
    Accuracy Results:
    {accuracy_data}
    
    Data (First 20 for size limits):
    {df.head(20).to_json(orient='records')}
    
    Return ONLY the raw HTML code without any markdown code blocks or wrapper text. Do not include ```html or ```.
    """

    print("Sending data to LLM for analysis and HTML generation...")
    try:
        response = groq_client.chat.completions.create(
            model='openai/gpt-oss-120b',
            messages=[{"role": "user", "content": prompt}]
        )
        
        html_content = response.choices[0].message.content
        
        # Clean up markdown if the LLM wrapped it
        if html_content.startswith("```html"):
            html_content = html_content[7:]
        if html_content.endswith("```"):
            html_content = html_content[:-3]
            
        # 3. Save to report.html
        with open("report.html", "w", encoding="utf-8") as f:
            f.write(html_content.strip())
            
        print("Successfully generated report.html!")
        
    except Exception as e:
        print(f"Error generating analysis: {e}")

if __name__ == "__main__":
    analyze_and_generate_html()
