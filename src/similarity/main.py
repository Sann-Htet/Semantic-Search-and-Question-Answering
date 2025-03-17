import logging
import os
from collections import defaultdict

from haystack import Answer
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader, PreProcessor
from haystack.pipelines import MostSimilarDocumentsPipeline
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.utils import print_answers
from rich.console import Console
from rich.layout import Layout
from rich.style import Style
from rich.text import Text

cleaner = PreProcessor(clean_empty_lines=True,clean_whitespace=True,split_by="word",split_respect_sentence_boundary=False)
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

document_store = InMemoryDocumentStore(embedding_dim=384)
bmr = BM25Retriever(document_store=document_store)
doc_dir = "../../data/dataset"


files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)
retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L12-v2", use_gpu=True,
)


query_index_pipline = TextIndexingPipeline(document_store)
doc = query_index_pipline.run("../../data/doc1.txt")
doc_id = doc.get("documents")[0].id
document_store.update_embeddings(retriever)
msd_pipeline = MostSimilarDocumentsPipeline(document_store)
msd_result = msd_pipeline.run(document_ids=[doc_id])
words =   cleaner.process(doc.get("documents"))[0]
document_store.get_document_by_id(id=doc_id)
# queries  = [txt for txt in words.content.split("\n") if txt]
queries = ["What is the score?","Who is the artist?","How is it recieved?","Name of the sountrack?"]
reader = FARMReader("sentence-transformers/all-MiniLM-L12-v2", use_gpu=True)
results = reader.predict_batch(queries, msd_result[0], 3)
print_answers(results)

console = Console(highlight=False, record=True)
# console.print(results)


doc_dics: defaultdict[str, list[dict]] = defaultdict(list[dict])

answers_list: list[list[Answer]] = results.get("answers")
css_highlight_colors = [
    "#FF7F00", # Orange
    "#FFFF22", # Light Purple
    "#F0C674", # Sand
    "#8DD7CF", # Sea Green
    "#FBBDFE", # Pink
    "#E0FFDA", # Yellow Green
    "#9B59B6", # Dark Pink
    "#FFC107",  # Dark Orange
]

query_answers: list[str,list[Answer]] = list(zip(queries,answers_list))
input_content = document_store.get_document_by_id(doc_id).content

for query,answers in query_answers:
    for ans in answers:
        ans_doc_id = ans[0].document_ids[0]
        doc_dics[ans_doc_id].append(dict(query=query,answer=ans[0]))



for d in doc_dics:
    question_texts = Text()
    answer_context = Text(end="\n\n")

    input_text = Text(input_content)
    compare_doc = Text(document_store.get_document_by_id(d).content)
    for i,qa in enumerate(doc_dics[d]):
        if i >= len(css_highlight_colors) :
            i = 0
        answer:Answer = qa["answer"]
        question_texts.append("query: ")
        question_texts.append(qa["query"],Style(color=css_highlight_colors[i]))
        question_texts.append("\n\n")
        answer_context.append(f" [score {answer.score}] ")
        answer_context.append(answer.context,Style(color=css_highlight_colors[i]))
        answer_context.append("\n\n")
        for span in qa["answer"].offsets_in_document:
            compare_doc.stylize(css_highlight_colors[i], span.start, span.end)
    layout = Layout()

    layout.split_column(Layout(name="input"),Layout(name="compare"),Layout(name="context"))
    layout["compare"].split_row( Layout(name="queries"),Layout(name="answer"))
    layout["input"].update(input_text)
    layout["context"].update(compare_doc)
    layout["queries"].update(question_texts)
    layout["answer"].update(answer_context)
    console.print(layout)


console.save_html("output.html")
