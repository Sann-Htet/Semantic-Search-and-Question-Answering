from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers

document_store = InMemoryDocumentStore(use_bm25=True,use_gpu=True,similarity="cosine")
retriever = BM25Retriever(document_store=document_store)

doc_dir = "data/similarity_search"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt14.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

got_docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
document_store.write_documents(got_docs)
document_store.update_bm25()

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)

prediction = pipe.run(
    query="Who is Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}},
)

print_answers(prediction)
