# PDF Question Answering App

A Streamlit application that allows users to upload a PDF document, create embeddings using Sentence Transformers, and answer questions based on the document's content using GPT-4.

## Features

- **PDF Upload**: Upload a PDF file for processing.
- **Text Extraction**: Extract text from the uploaded PDF.
- **Text Chunking**: Split the text into manageable chunks.
- **Embedding Generation**: Generate embeddings for the text chunks using a pre-trained model.
- **Querying**: Retrieve the most relevant chunks for a given question.
- **Answer Generation**: Use GPT-4 to generate answers based on the retrieved context.

## Installation

```bash
   git clone https://github.com/yourusername/pdf-question-answering.git
   cd pdf-question-answering
   pip install -r requirements.txt
```
## Running the App

1. Set your OpenAI API key as an environment variable:

```bash

export OPENAI_API_KEY='your-api-key'

```

2. Run the Streamlit app:

```bash

streamlit run app.py

```

3. Access the app at `http://localhost:8501` in your web browser.

## Usage

- **Upload PDF**: Use the file uploader to upload your PDF document.

- **Embedding Generation**: Embedding model here uses CPU so it might take 4-5 mins for handbook.pdf to process.

- **Enter Questions**: Input your questions in the text area, one per line.

- **Get Answers**: Click the "Get Answers" button to retrieve answers.

- **Download Answers**: Download the answers in JSON format.



## Future Improvements to agent

- Processing tables, images, and varied fonts as well.

- Better chunking(eg: overlap chunking)

- Speed up embedding generation for large PDFs using parallel processing(GPU).

- Add tools to agents depending on the use case and type of pdf/questions asked.
