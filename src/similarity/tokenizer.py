import logging
import os

import spacy
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever, PreProcessor
from haystack.pipelines import MostSimilarDocumentsPipeline
from haystack.pipelines.standard_pipelines import TextIndexingPipeline

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_md")
cleaner = PreProcessor(clean_empty_lines=True,clean_whitespace=True,split_by="passage",split_respect_sentence_boundary=False)
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)


document_store = InMemoryDocumentStore(embedding_dim=384)
doc_dir = "/home/nozander/Workspace/doc-similar/data/dataset"


files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)
retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L12-v2", use_gpu=True,
)


query_index_pipline = TextIndexingPipeline(document_store)
doc = query_index_pipline.run("/home/nozander/Workspace/doc-similar/data/doc1.txt")
doc_id = doc.get("documents")[0].id
document_store.update_embeddings(retriever)
msd_pipeline = MostSimilarDocumentsPipeline(document_store)
msd_result = msd_pipeline.run(document_ids=[doc_id])
words =   cleaner.process(doc.get("documents"))[0]
list_of_tokens  = [list(nlp(txt)) for txt in words.content.split("\n") if txt]

# queries = ["What is the score?","Who is the artist?","How is it recieved?","Name of the sountrack?"]
# reader = FARMReader("sentence-transformers/all-MiniLM-L12-v2", use_gpu=True)
# results = reader.predict_batch(queries, msd_result[0], 3)
# print_answers(results)
