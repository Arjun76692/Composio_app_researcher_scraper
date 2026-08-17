# Composio App Researcher

An autonomous agent pipeline that researches developer apps using the Composio SDK and Groq LLMs to determine authentication methods, access models, and API surfaces.

## How to run the agent (7 Steps)

1. **Clone the repository** and navigate into the project folder.
2. **Install the dependencies** by setting up a Python environment and running: 
   `pip install pandas python-dotenv groq composio-core pydantic`
3. **Configure your API keys:** Create a `.env` file in the root directory and add your credentials:
   ```env
   GROQ_API_KEY=your_groq_key
   COMPOSIO_API_KEY=your_composio_key
   ```
4. **Provide the input list:** Ensure the target 100 apps are listed in `apps.csv` with `App`, `Website`, and `Category` columns.
5. **Run the autonomous research agent:** Execute `python agent.py`. The agent will search the web using Composio's tools, extract the technical data, and save its progress continuously to `research_results_checkpoint.csv`.
6. **Verify the data:** Run `python verify_accuracy.py` to extract a weighted sample of low-confidence rows and automatically cross-check them against Composio's live toolkit catalog.
7. **View the final results:** Open the `report.html` file in your browser to view the final synthesized data, verification metrics, and system constraints.

*Note: The agent is built with smart checkpointing. If you hit an API rate limit, simply wait and re-run `python agent.py`. It will read the checkpoint file and automatically resume where it left off without duplicating research.*
