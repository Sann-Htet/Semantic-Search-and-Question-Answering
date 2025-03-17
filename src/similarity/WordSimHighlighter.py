import spacy
import torch
from torch.nn.functional import cosine_similarity
from collections import defaultdict

def generate_similar_phrases(query_text, compare_text, retriever, score=0.55):
    """
    Args:
        query_text (str): The input query text.
        compare_text (str): The text to compare word similarity.
        score (float): Similarity threshold for considering phrases as similar (default: 0.55).

    Returns:
        dict: A dictionary where keys are query phrases, and values are lists of similar document phrases.
    """
    spacy_model = spacy.load("en_core_web_md")

    pos_tags = ["NOUN", "ADJ", "VERB"]

    query = spacy_model(query_text)
    extracted_doc = spacy_model(compare_text)

    similar_phrases = defaultdict(list)

    for pos_tag in pos_tags:
        if pos_tag == "NOUN":
            query_pos_tags = [str(chunk) for chunk in query.noun_chunks]
            extracted_doc_pos_tags = [str(chunk) for chunk in extracted_doc.noun_chunks]
        else:
            query_pos_tags = [str(chunk) for chunk in query if chunk.pos_ == pos_tag]
            extracted_doc_pos_tags = [str(chunk) for chunk in extracted_doc if chunk.pos_ == pos_tag]

        # Check if the lists are not empty before creating tensors
        if query_pos_tags and extracted_doc_pos_tags:
            # Get similarity scores and filter for scores above 0.5
            query_pos_tags_embed = torch.tensor(retriever.embed_queries(query_pos_tags))
            extracted_doc_pos_tags_embed = torch.tensor(retriever.embed_queries(extracted_doc_pos_tags))
            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(query_pos_tags_embed.unsqueeze(1), extracted_doc_pos_tags_embed.unsqueeze(0), dim=-1)

            # Find indices where similarity is greater than 0.55
            indices = torch.nonzero(similarity_matrix > score, as_tuple=False)

            # Gather the results based on the indices
            for index in indices:
                query_phrase = query_pos_tags[index[0]]
                doc_phrase = extracted_doc_pos_tags[index[1]]

                similar_phrases[query_phrase].append(doc_phrase)
        
    return {key: list(set(value)) for key, value in similar_phrases.items()}