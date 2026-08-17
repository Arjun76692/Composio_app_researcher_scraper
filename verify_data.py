import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def verify_results():
    print("Starting Verification Loop...")
    try:
        df = pd.read_excel("research_results.xlsx")
    except Exception as e:
        print("Could not load research_results.xlsx. Run agent.py first.")
        return

    # Sample 3 random apps to verify
    sample_df = df.sample(min(3, len(df)))
    
    for index, row in sample_df.iterrows():
        app_name = row['App']
        print(f"\nVerifying: {app_name}")
        
        # We prompt the LLM to read the row and decide if the data makes sense
        prompt = f"""
        You are a Data Quality Auditor. Review this data for {app_name}:
        Category: {row['category']}
        Auth Methods: {row['auth_methods']}
        Self Serve vs Gated: {row['self_serve_vs_gated']}
        API Surface: {row['api_surface']}
        Buildability: {row['buildability_verdict']}
        
        Is this information logically consistent and generally accurate based on your knowledge of the app {app_name}?
        Answer concisely with 'YES' or 'NO' and a 1-sentence reason.
        """
        
        try:
            response = groq_client.chat.completions.create(
                model='openai/gpt-oss-120b',
                messages=[{"role": "user", "content": prompt}]
            )
            print(f"Audit Result: {response.choices[0].message.content.strip()}")
        except Exception as e:
            print(f"Error checking {app_name}: {e}")

if __name__ == "__main__":
    verify_results()
