# MovieQuoteGPT

MovieQuoteGPT is a sarcastic, movie-quoting chatbot built by fine-tuning the **TinyLlama-1.1B-Chat-v1.0** model. The project features an end-to-end pipeline, starting from AI-driven dataset preparation to fine-tuning the model using LoRA under different quantization levels (2-bit, 4-bit, and 8-bit). It also includes a detailed performance analysis and an interactive Gradio web interface.

## Features

- **Automated Dataset Generation:** Uses the Groq API (`llama-3.3-70b-versatile`) to automatically generate conversational questions that perfectly map to iconic movie quotes.
- **Efficient Fine-Tuning:** Implements Parameter-Efficient Fine-Tuning (PEFT) using LoRA to train the chatbot on consumer hardware.
- **Multi-Level Quantization:** Fine-tunes and evaluates the model across 2-bit, 4-bit, and 8-bit quantization settings using `bitsandbytes`.
- **Comprehensive Performance Analysis:** Compares model size, inference speed (tokens/sec), output quality, and training time across the different quantization levels, generating detailed visualizations (bar charts, scatter plots, radar charts).
- **Interactive Web UI:** Features a fully functional Gradio interface allowing users to chat with the bot and dynamically switch between the 2-bit, 4-bit, and 8-bit quantized models.

## Project Structure

- `MovieQuoteGPT_Dataset_Preparation.py`: Script used to process raw movie quotes and generate corresponding natural language questions using the Groq API. Outputs the final training dataset.
- `MovieQuoteGPT_Training_Quantization_Chatbot.py`: The core script containing data preparation, model tokenization, quantized LoRA fine-tuning, performance evaluation metrics, visualization generation, and the Gradio web UI.
- `movie_quotes_with_questions_trimmed (1).csv`: The dataset containing the movie quotes and their corresponding AI-generated questions used for fine-tuning.

## Tech Stack

- **Model:** TinyLlama 1.1B Chat
- **Libraries:** PyTorch, Hugging Face Transformers, PEFT (LoRA), BitsAndBytes, Datasets, Gradio, Pandas, Matplotlib, Instructor
- **APIs:** Groq API (for dataset generation)

## Key Insights on Quantization Trade-offs

Through comprehensive testing, the project reveals interesting insights into quantization:
- **2-bit Quantization:** Highly compact and suitable for extreme memory constraints while surprisingly maintaining a high output quality score.
- **4-bit Quantization:** Offers the fastest training time but presented the most significant degradation in output quality in this specific setup.
- **8-bit Quantization:** Delivers the best inference speed and excellent quality, ideal for environments with fewer memory restrictions.

## Usage

To run the fine-tuning and chatbot pipeline, install the required dependencies:

```bash
pip install transformers accelerate bitsandbytes datasets peft gradio pandas torch matplotlib
```

Then, execute the training and chatbot script:

```bash
python MovieQuoteGPT_Training_Quantization_Chatbot.py
```

This will train the models, generate the performance analysis charts, and launch the interactive Gradio interface locally.
