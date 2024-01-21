import os
from datetime import datetime
from arxivscraper import Scraper
import fitz  # PyMuPDF
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def extract_metadata(pdf_path):
    with fitz.open(pdf_path) as pdf_document:
        metadata = {
            'title': pdf_document.metadata['title'],
            'author': pdf_document.metadata['author'],
            'subject': pdf_document.metadata['subject'],
            'created_date': pdf_document.metadata['created_date'],
            'modified_date': pdf_document.metadata['modified_date']
            # add more metadata fields if needed
        }
        return metadata

def process_pdf(pdf_path, output_folder):
    # extract metadata
    metadata = extract_metadata(pdf_path)

    # extract text content
    with fitz.open(pdf_path) as pdf_document:
        text_content = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text_content += page.get_text()

    # save metadata and text content to a CSV file
    csv_filename = os.path.join(output_folder, 'papers_metadata.csv')
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
            fieldnames = list(metadata.keys()) + ['text_content']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    with open(csv_filename, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        metadata['text_content'] = text_content
        writer.writerow(metadata)

def fetch_all_arxiv_papers(start_date, end_date, output_folder):
    categories = ['cs', 'math', 'physics', 'quant-ph']  # add more categories as needed

    for category in categories:
        scraper = Scraper(category=category, date_from=start_date, date_until=end_date)
        output = scraper.scrape()

        for paper in output:
            title = paper['title'].replace('\n', ' ').strip()
            pdf_url = paper['pdf_url']

            # download the PDF
            pdf_path = os.path.join(output_folder, f'{title}.pdf')
            os.system(f'wget -O "{pdf_path}" {pdf_url}')

            # process the downloaded PDF
            process_pdf(pdf_path, output_folder)

def train_language_model(output_folder, model_name='gpt2', num_train_epochs=1):
    # load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # load the processed dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=os.path.join(output_folder, 'papers_metadata.csv'),
        block_size=128,
    )

    # data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # training arguments
    training_args = TrainingArguments(
        output_dir=output_folder,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # train the model
    trainer.train()

    # save the trained model
    model.save_model(os.path.join(output_folder, 'trained_model'))

if __name__ == "__main__":
    # specify the desired start and end dates
    start_date = datetime(1991, 8, 14)  # arXiv launch date. change if needed.
    end_date = datetime.now()

    # specify the folder where PDFs, metadata, and the model will be saved
    output_folder = './arxiv_papers'

    # create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Uncomment and run the following line to fetch, process, and train a language model on papers from arXiv.
    # Make sure you have the bandwidth and storage for this.
    # TL;DR Don't run this.
    # fetch_all_arxiv_papers(start_date, end_date, output_folder)
    train_language_model(output_folder, model_name='gpt2', num_train_epochs=1)
