from flask import Flask, render_template, url_for, request, redirect, jsonify, make_response, flash
import torch
from datasets import Dataset
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever, PromptNode, PromptTemplate, DensePassageRetriever,  FARMReader
from haystack.pipelines import GenerativeQAPipeline, Pipeline,  DocumentSearchPipeline, ExtractiveQAPipeline
from haystack import Label, Answer, Document
from datasets import load_dataset
import torch
import gc

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sohai'
document_store = None

@app.route("/")
@app.route("/searcher/<query>/<answers>/")
def searcher(query=None, answers=None):
    url_for('static', filename='style.css')
    return render_template("index.html", query=query, answers=answers)

@app.route('/search', methods=['GET', 'POST'])
def search_request():
    retriever = request.form.get('select')
    reader_generator = request.form.get('select-x')
    query = request.form.get('query')

    if retriever == 'BM25':
        retriever_model = BM25Retriever(document_store=document_store, top_k=2)
    elif retriever == 'DPR':
        retriever_model = DensePassageRetriever(document_store=document_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            embed_title=False)
        document_store.update_embeddings(retriever=retriever_model)
        
    if reader_generator == 'MiniLM':
        reader_generator_model = load_reader(1)
    elif reader_generator == 'RoBERTa':
        reader_generator_model = load_reader(2)
    elif reader_generator == 'ALBERT':
        reader_generator_model = load_reader(3)
    else:
        reader_generator_model = load_generator()

    # create pipeline
    if reader_generator == 'Flan-T5':
        pipeline = generative_pipeline(retriever_model, reader_generator_model)
    else:
        pipeline = extractive_pipeline(reader_generator_model, retriever_model)
    
    preds = pipeline.run(query=query)

    del pipeline, reader_generator_model, retriever_model
    gc.collect()
    torch.cuda.empty_cache()

    answers = ""
    for pred in preds["answers"]:
        answers += "||" + pred.answer + "||" + str(pred.score)

    print(answers)
    return redirect(url_for('searcher', query=query, answers=answers))

def get_dataset():
    subjqa = load_dataset("subjqa", "tripadvisor")
    dfs = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}
    for index, row in dfs["train"].iterrows():
        if row["answers.text"].size == 0:
            dfs["train"].drop(axis=0, index=index, inplace=True)
    return dfs

def load_document_store():
    document_store = ElasticsearchDocumentStore(similarity="dot_product",embedding_dim=768)
    if len(document_store.get_all_documents()) or len(document_store.get_all_labels()) > 0:
        document_store.delete_documents("document")
        document_store.delete_documents("label")
    dfs = get_dataset()
    for split, df in dfs.items():
        # Exclude duplicate reviews
        docs = [{"content": row["context"], 
                "meta":{"item_id": row["title"], "question_id": row["id"], 
                        "split": split}} 
            for _,row in df.drop_duplicates(subset="context").iterrows()]
        document_store.write_documents(documents=docs, index="document")
    print(f"Loaded {document_store.get_document_count()} documents")
    return document_store

def load_generator():
    lfqa_prompt = PromptTemplate(
    name="lfqa",
    prompt_text="""Synthesize a comprehensive answer from the following text for the given question. 
                             Provide a clear and concise response that summarizes the key points and information presented in the text. 
                             Your answer should be in your own words and be no longer than 50 words. 
                             \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
    )
    return PromptNode("google/flan-t5-large", default_prompt_template=lfqa_prompt)

def load_reader(option=1):
    if option == 1:
        return FARMReader("../minilm_squad_subjqa")
    elif option == 2:
        return FARMReader("../roberta_squad_subjqa")
    else:
        return FARMReader("../albert_squad_subjqa")    
    
def extractive_pipeline(reader, retriever):
    return ExtractiveQAPipeline(reader, retriever)

def generative_pipeline(retriever, generator):
    pipe = Pipeline()
    pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
    pipe.add_node(component=generator, name="prompt_node", inputs=["retriever"])
    return pipe

if __name__ == "__main__":
    document_store = load_document_store()
    app.debug = True
    app.run(host='0.0.0.0', port=5000)