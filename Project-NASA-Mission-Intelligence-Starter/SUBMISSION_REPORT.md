# NASA Mission Intelligence - Submission Report

## Project Status

All implementation TODOs in the Python project files have been completed. The completed system includes:

- OpenAI chat response generation with retrieved NASA mission context
- ChromaDB backend discovery and RAG retrieval
- Text ingestion, chunking, metadata extraction, stable document IDs, and ChromaDB persistence
- Streamlit chat interface integration
- RAGAS response evaluation integration
- Batch evaluation with a mission-relevant test dataset
- Vocareum OpenAI endpoint support for course API keys

## End-to-End Test Summary

The full embedding workflow was run successfully against the provided NASA text corpus.

Results:

- Files processed: 12
- Total chunks created: 17,657
- Documents added to ChromaDB: 17,657
- Errors: 0
- ChromaDB collection: `nasa_space_missions_text`
- ChromaDB directory: `chroma_db_openai`

Mission breakdown:

- Apollo 11: 6 files, 9,598 chunks
- Apollo 13: 3 files, 7,106 chunks
- Challenger: 3 files, 953 chunks

The Streamlit app was launched successfully at:

```text
http://127.0.0.1:8501
```

The retrieval workflow was tested with:

```text
What happened during Apollo 13?
```

The system returned relevant Apollo 13 document excerpts from the ChromaDB collection.

Batch evaluation support was added with:

- Dataset: `test_questions.json`
- Evaluation script: `batch_evaluate.py`
- Default output: `batch_evaluation_results.json`

The dataset includes 6 mission-relevant questions spanning overview, emergency, disaster analysis, crew, technical, and timeline categories.

## Challenges Faced and Solutions

### Vocareum API Key Support

The provided course key uses the `voc-` prefix and is not accepted by the default OpenAI API endpoint. The course notebooks showed that Vocareum keys require:

```text
https://openai.vocareum.com/v1
```

Solution:

- Added automatic detection of `voc-` keys
- Routed LLM calls, embedding calls, Chroma embedding functions, and RAGAS evaluators through the Vocareum base URL when needed

### Missing Optional RAGAS Metric Dependencies

RAGAS evaluation required optional packages that were not installed by default.

Solution:

- Installed and added `sacrebleu` for BLEU evaluation
- Installed and added `rouge_score` for ROUGE evaluation
- Updated `requirements.txt` so future installs include these dependencies

### Data Directory Mismatch

The README described a `data/` directory, but the project files were stored in `data_text/`.

Solution:

- Updated the embedding pipeline to automatically use `data_text/` when mission folders are not found directly under the provided base path

### Streamlit First-Run Prompt and Local Port Binding

Streamlit initially prompted for usage statistics and needed local server permissions.

Solution:

- Launched Streamlit with usage stats disabled
- Confirmed the app responds locally with HTTP 200

## Additional Features and Improvements

- Stable document IDs based on mission, source, and chunk index
- Re-run support with `skip`, `update`, and `replace` modes
- Metadata-rich chunks with mission, source, file type, category, chunk index, and timestamps
- Mission-aware retrieval filtering support
- Mission filter control in the Streamlit sidebar
- Deduplication of repeated retrieved snippets before context construction
- Context formatting with clear source labels
- Defensive RAGAS error handling so evaluation issues do not crash the chat app
- BLEU and ROUGE fallback scoring using retrieved context as reference for live chat evaluation
- Dependency compatibility pin for `langchain-community`

## Sample Queries and Expected Responses

### Query 1

```text
What happened during Apollo 13?
```

Expected response:

The assistant should summarize the Apollo 13 mission incident using retrieved Apollo 13 transcript context. A good response should mention that Apollo 13 experienced a serious in-flight emergency, Mission Control worked with the crew, and the mission became focused on safely returning the crew to Earth.

### Query 2

```text
What were the main events of the Apollo 11 mission?
```

Expected response:

The assistant should describe Apollo 11 as the first crewed Moon landing mission and mention major phases such as launch, lunar module operations, Moon landing, surface activity, and return.

### Query 3

```text
What did the Apollo 11 flight plan describe?
```

Expected response:

The assistant should explain that the flight plan contains scheduled mission activities, procedures, spacecraft operations, crew tasks, and timing information.

### Query 4

```text
What communications happened between Mission Control and Apollo 13?
```

Expected response:

The assistant should use transcript context to summarize communications between the Apollo 13 crew and Mission Control, including status updates, instructions, and troubleshooting guidance.

### Query 5

```text
What happened during the Challenger STS-51L mission audio transcripts?
```

Expected response:

The assistant should retrieve Challenger transcript context and summarize the relevant mission audio content. If the retrieved context is limited, it should say that the answer is based only on the available transcript excerpts.

### Query 6

```text
Compare Apollo 11 and Apollo 13 based on the mission documents.
```

Expected response:

The assistant should compare Apollo 11 as a successful lunar landing mission with Apollo 13 as a mission dominated by emergency response and crew survival, citing differences in mission goals, operations, and communications.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the ChromaDB database:

```bash
python embedding_pipeline.py --openai-key "$OPENAI_API_KEY" --data-path .
```

Run the app:

```bash
streamlit run chat.py
```

Run batch evaluation:

```bash
python batch_evaluate.py --questions test_questions.json --chroma-dir chroma_db_openai --collection-name nasa_space_missions_text
```

This command loads the test questions, retrieves mission-filtered context, generates answers, computes evaluation scores, and writes aggregate results to `batch_evaluation_results.json`.
