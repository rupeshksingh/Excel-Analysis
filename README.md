# 🤖 PandasAI Data Analysis Streamlit App

## Overview

This Streamlit application leverages PandasAI and OpenAI's language models to provide intelligent data analysis on Excel files.

## Features

- Multi-file Excel upload

- AI-powered data querying

- Clarification questions generation

- Context-aware query modification

- Explanation of analysis process

## Setup Instructions

### Prerequisites

- Python 3.8+

- OpenAI API Key

### Installation

1\. Clone the repository

2\. Create a virtual environment

3\. Install dependencies:

   `
   pip install -r requirements.txt
   `

### Configuration

1\. Create a `.env` file

2\. Add your OpenAI API Key:

   `
   OPENAI_API_KEY=your_openai_api_key_here
   `

### Running the App

`
streamlit run app.py
`

## Usage

1\. Upload Excel files

2\. Enter a query about your data

3\. Answer or skip clarification questions

4\. Process and view results

## Note

Ensure your Excel files are structured consistently for best results.
