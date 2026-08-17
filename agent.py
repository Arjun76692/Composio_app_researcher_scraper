import os
import time
import json
import csv
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import Groq
from composio import Composio

# Load environment variables
load_dotenv()

# Initialize Groq Client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize Composio Client (v0.19.0+)
composio_client = Composio(api_key=os.getenv("COMPOSIO_API_KEY"))

class NoSearchDataError(Exception):
    pass

class AppResearchResult(BaseModel):
    category: str = Field(description="Category and what it does in one line.")
    auth_methods: str = Field(description="Auth method(s): OAuth2, API key, Basic, token, or other.")
    self_serve_vs_gated: str = Field(description="Self-serve vs gated: can a developer get credentials themselves for free or on a trial, or does it need a paid plan, admin approval, or a partnership / contact-sales gate.")
    api_surface: str = Field(description="API surface: documented public REST / GraphQL, roughly how broad. Explicitly state whether it can be an MCP server or agent-callable skills.")
    buildability_verdict: str = Field(description="Buildability verdict: could this be an agent toolkit today, and the main blocker if not.")
    evidence: str = Field(description="Evidence: MUST BE ONLY the docs URLs from the provided text that back up your answers.")
    confidence: str = Field(description="Confidence: 'High', 'Medium', or 'Inferred'. If data is completely missing and you made a best-effort guess, mark as Inferred.")
    needs_human_review: bool = Field(default=False, description="Set to True ONLY if the provided text lacks almost all necessary information, meaning a human needs to manually review it.")

def execute_exa_search(query: str, user_id: str, official_domain: str = None) -> str:
    """Executes a Composio EXA search based on Groq's dynamic tool call."""
    try:
        response = composio_client.tools.execute(
            "EXA_SEARCH",
            arguments={"query": query, "num_results": 3},
            user_id=user_id,
            dangerously_skip_version_check=True
        )
        
        parsed_results = []
        data = response
        if hasattr(response, 'data'): data = response.data
        elif hasattr(response, 'results'): data = response.results
        if isinstance(data, str):
            try: data = json.loads(data)
            except: pass
            
        results_list = []
        if isinstance(data, dict):
            if "results" in data: results_list = data["results"]
            elif "data" in data and isinstance(data["data"], dict) and "results" in data["data"]: results_list = data["data"]["results"]
            elif "data" in data and isinstance(data["data"], list): results_list = data["data"]
        elif isinstance(data, list):
            results_list = data
            
        for item in results_list:
            if isinstance(item, dict):
                url = item.get('url', '')
                # Deprioritize noisy non-doc sources
                is_low_quality = any(bad in url.lower() for bad in [
                    'lever.co', 'jobs.', 'stackoverflow.com', '/blog/', 'medium.com'
                ])
                is_official = official_domain and official_domain in url.lower()
                tag = " [OFFICIAL DOMAIN]" if is_official else (" [LOW QUALITY/UNOFFICIAL]" if is_low_quality else "")
                parsed_results.append(
                    f"Title: {item.get('title', '')}{tag}\nURL: {url}\n"
                    f"Snippet: {str(item.get('text', item.get('snippet', item.get('content', ''))))[:2000]}"
                )
                
        if not parsed_results:
            return "No results found for this query."
            
        return "\n\n".join(parsed_results)
    except Exception as e:
        return f"Search failed: {str(e)}"

def agentic_research_loop(app_name: str, website: str, user_id: str) -> str:
    """Uses Groq natively with tool calling to autonomously research the app."""
    messages = [
        {
            "role": "system", 
            "content": f"You are an autonomous research agent investigating the developer API for '{app_name}' ({website}). "
                       f"Your goal is to find information on: 1) The PRIMARY/CORE API auth methods (not just newer MCP features), "
                       f"2) Gated vs Self-serve API access for the core platform, 3) API surface (REST/GraphQL) breadth, "
                       f"and 4) whether an official MCP server exists (as a separate, additional finding, not a replacement for #1-3). "
                       f"IMPORTANT: Research the company's main/flagship API first — do not let a niche or newer product feature "
                       f"(like a beta MCP server) dominate your answer if the company has a broader, more established core API. "
                       f"Use the 'exa_search' tool to search the web. PREFER searching the official site "
                       f"first, e.g. 'site:{website} API documentation authentication' before trying general queries. "
                       f"Once you have gathered enough information OR tried searching 3 times, output a final comprehensive summary of your findings including all the source URLs you found. "
                       f"CRITICAL: You ONLY have access to the 'exa_search' tool. Do NOT attempt to call any other tools like 'open_file', 'open', or 'read'. Just use exa_search."
        }
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "exa_search",
                "description": "Search the web for API documentation, developer portals, MCP servers, or pricing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query (e.g., 'HubSpot API authentication methods')."}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    # Run the agent loop (max 4 iterations to prevent infinite loops)
    for step in range(4):
        try:
            response = groq_client.chat.completions.create(
                model='openai/gpt-oss-120b',
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1
            )
        except Exception as e:
            error_msg = str(e)
            if "tool call validation failed" in error_msg.lower():
                print(f"    [Agent Correction] Model hallucinated a tool. Correcting...")
                messages.append({
                    "role": "user",
                    "content": f"You attempted to use an invalid tool. ERROR: {error_msg}. Reminder: You ONLY have access to the 'exa_search' tool."
                })
                continue
            raise e
            
        message = response.choices[0].message
        messages.append(message)
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "exa_search":
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query", app_name)
                    print(f"    [Agent Decision] Searching web for: '{query}'")
                    
                    search_result = execute_exa_search(query, user_id, official_domain=website)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "exa_search",
                        "content": search_result
                    })
                    time.sleep(2.5) # Respect Groq rate limits
        else:
            # The agent is done searching and has provided a final summary!
            print("    [Agent Decision] Research complete. Extracting final data...")
            return message.content
            
    # If we hit max steps, just return the last message
    return messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

def extract_data_with_groq(app_name: str, website: str, agent_summary: str) -> dict:
    """Uses Groq to extract the final JSON schema from the agent's summary."""
    schema = AppResearchResult.model_json_schema()
    
    prompt = f"""
    You are an expert AI integrations researcher.
    We are researching the app '{app_name}' (website: {website}).
    
    Below is the summary compiled by our autonomous research agent:
    ---
    {agent_summary}
    ---
    
    Based on the text above, extract the required information.
    - If some information is completely missing, make a best-effort guess based on industry standards, set confidence to "Inferred".
    - If the text is completely useless and you cannot infer anything, set needs_human_review to True.
    - For the 'evidence' field, STRONGLY PREFER URLs tagged [OFFICIAL DOMAIN] over unofficial 
      community repos, blogs, job postings, or third-party wrapper tools. Only cite an unofficial 
      source (like a GitHub community MCP wrapper) if NO official documentation was found, and if 
      you do, explicitly note in buildability_verdict that no official docs exist yet.
    - Never cite job listings, hiring pages, or Stack Overflow as evidence for auth or API details.
    
    Return a JSON object that strictly follows this schema:
    {json.dumps(schema, indent=2)}
    """
    
    response = groq_client.chat.completions.create(
        model='openai/gpt-oss-120b',
        messages=[{"role": "system", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    raw = json.loads(response.choices[0].message.content)
    
    # Validate against the pydantic schema instead of trusting raw dict
    try:
        validated = AppResearchResult(**raw)
        data = validated.model_dump()
        
        # Safety net: if core fields are empty, force human review regardless of what model said
        core_fields = [data['category'], data['auth_methods'], data['api_surface']]
        if all(not f or f.strip() == "" for f in core_fields):
            data['needs_human_review'] = True
            data['confidence'] = "Low"
            data['buildability_verdict'] = "Needs human review: extraction returned empty core fields"
            
        return data
    except Exception as e:
        print(f"  -> Schema validation failed for {app_name}: {e}")
        raise  # let the retry loop in main() catch this and retry

def main():
    print("Starting Composio Agentic Researcher Pipeline...")
    
    df = pd.read_csv("apps.csv")
    
    print("Resolving Composio EXA connection user_id...")
    accounts = composio_client.connected_accounts.list().items
    user_id = "default"
    for acc in accounts:
        acc_dict = acc.model_dump() if hasattr(acc, 'model_dump') else dict(acc)
        if acc_dict.get('toolkit', {}).get('slug') == 'exa':
            user_id = acc_dict.get('user_id')
            break
    print(f"Using user_id: {user_id}")
    
    output_file = "research_results_checkpoint.csv"
    headers = ["App", "Website", "category", "auth_methods", "self_serve_vs_gated", "api_surface", "buildability_verdict", "evidence", "confidence", "needs_human_review"]
    
    if not os.path.exists(output_file):
        with open(output_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    # Resume-skip: only skip apps that completed SUCCESSFULLY (not Low/human-review)
    completed_apps = set()
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            good_df = existing_df[existing_df['confidence'] != 'Low']
            bad_df = existing_df[existing_df['confidence'] == 'Low']
            completed_apps = set(good_df['App'].tolist())
            
            if len(bad_df) > 0:
                print(f"Purging {len(bad_df)} failed rows (will retry): {bad_df['App'].tolist()}")
                # Rewrite CSV with only the good rows
                good_df.to_csv(output_file, index=False)
            
            print(f"Resuming: {len(completed_apps)} apps already completed, skipping them.")
        except Exception:
            pass  # Empty or corrupt file, start fresh
    
    total = len(df)
    
    for index, row in df.iterrows():
        app_name = row['App']
        website = row['Website']
        
        if app_name in completed_apps:
            print(f"\n[{index+1}/{total}] Skipping {app_name} (already in checkpoint).")
            continue
        
        print(f"\n[{index+1}/{total}] Agent researching: {app_name}...")
        
        max_retries = 3
        extracted_data = None
        
        for attempt in range(max_retries):
            try:
                # Phase 1: Agentic Loop (Search + Reason)
                agent_summary = agentic_research_loop(app_name, website, user_id)
                
                # Phase 2: JSON Extraction
                extracted_data = extract_data_with_groq(app_name, website, agent_summary)
                break
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "rate limit" in error_msg.lower():
                    print(f"  -> Rate limit hit (429). Sleeping for 60 seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(60)
                else:
                    print(f"  -> Error processing {app_name}: {e}. (Attempt {attempt+1}/{max_retries})")
        
        if extracted_data:
            final_row = {"App": app_name, "Website": website, **extracted_data}
        else:
            print(f"  -> Exhausted retries for {app_name}. Flagging for human review.")
            final_row = {
                "App": app_name, "Website": website, "category": "", "auth_methods": "", "self_serve_vs_gated": "",
                "api_surface": "", "buildability_verdict": "Needs human review: Agent got stuck finding or parsing API docs",
                "evidence": "", "confidence": "Low", "needs_human_review": True
            }
            
        with open(output_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(final_row)
            
        completed_apps.add(app_name)
        time.sleep(2.5)
        
    print(f"\nPipeline Complete! Results saved incrementally to '{output_file}'.")

if __name__ == "__main__":
    main()
