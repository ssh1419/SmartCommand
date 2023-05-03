import pickle 
import json
import numpy 

#%%
def semantic_search_sbert(query:str, command_embeddings, title_embeddings, model:str):
    """Get Max score between score of command and titles using 
    
    num_array is already sorted 
    """

    import sentence_transformers
    
    embedder = sentence_transformers.SentenceTransformer(model)
    query_embeddings = embedder.encode(query, 
                                      convert_to_numpy=True, 
                                      normalize_embeddings=True)
    
    command_scores = sentence_transformers.util.semantic_search(query_embeddings,
                                                      command_embeddings,
                                                      top_k=20000, # this k needs to be very big 
                                                      score_function=sentence_transformers.util.dot_score
                                                      )[0] # index 0 because we have only one query 

    title_scores = sentence_transformers.util.semantic_search(query_embeddings,
                                                      title_embeddings,
                                                      top_k=20000, # this k needs to be very big 
                                                      score_function=sentence_transformers.util.dot_score
                                                      )[0] # index 0 because we have only one query 
    
    # The order of scores are sorted by decreasing cosine similartiy scores automatically. 
    # So to compare the scores of commands and titles. It will be sorted by the corpus id
    command_scores = (sorted(command_scores, key=lambda d: d['corpus_id']) )
    title_scores = (sorted(title_scores, key=lambda d: d['corpus_id']) )
    
    
    max_scores = []
    for command_score, title_score in zip(command_scores, title_scores):
        corpus_id = command_score['corpus_id']
        max_score = max(command_score['score'], title_score['score'])
        max_scores.append({'corpus_id': corpus_id, 'score': max_score})

    # Sort the score by decreasing cosine similrty scores again
    max_scores = (sorted(max_scores, key=lambda d: d['score'], reverse=True) )
        
    return max_scores

# %% 
def semantic_search(query:str, command_embeddings, title_embeddings, method:str, model:str):
    """Return commands machine a query 

    Depending on the approach, the embeddings are of different shapes. 
    For SBERT, the embedding is at the sentence level. Hence, 
    the embeddings are 2D arrays, with axis-0 for each command. 
    For BERTScore, the embedding is at the token level. Hence, the 
    embeddings are 3D arrays, with axis 0 for each command, and 
    axis 1 for each token. 

    command_dict_list: a list of commands, 
                    each a dict: 
                {"key":str, "command":str, "when":str, "to-ebd": str}
                to-ebd is the string to embed the command itself or its label

    """

    if method == "sbert":
        scores = semantic_search_sbert(query, command_embeddings, title_embeddings, model)

    return scores 

#%%
def filter_results(num_array: numpy.ndarray, k:int, p:float):
    """Filter results using a combination of top-k and top-p.

    num_array is already sorted 
    """
    cum_sum = numpy.cumsum(num_array)
    total_score = numpy.sum(num_array)
    cum_sum = cum_sum / total_score
    top_p = numpy.where(cum_sum >= p)[0][0]
    print ("p cutoff index: ", top_p)
    index = min(k, top_p)
    return index

def combine_results(scores:dict[list], command_dict_list, k, p):
    """Combine results from semantic search of commands and titles with command dict list with filters,
    such as top-p. 

    scores: list of dicts, each dict has keys: corpus_id and score. 
    """
    results = []

    cutoff_index = filter_results([s['score'] for s in scores], k, p)

    for score in scores[:cutoff_index]:
        command_id = score['corpus_id']
        command = command_dict_list[command_id]
        results.append((command, score['score']))
    return results

# %%
def load_embeddings(pickle_file):
    with open(pickle_file, 'rb') as f:
        command_embeddings = pickle.load(f)
    return command_embeddings
#%% 
def main(query, command_embedding_pickle, method, model, k, p):

     # load 
    embeddings = load_embeddings(command_embedding_pickle)

    command_dict_list = embeddings['command_id']
    command_embeddings = embeddings['command_id_embeddings']
    title_embeddings = embeddings['command_title_embeddings']

    # search 
    scores = semantic_search(query, command_embeddings, title_embeddings, method, model)

    # combine 
    results = combine_results(scores, command_dict_list, k, p)

    return results

if __name__ == "__main__":
    import sys, os
    query = sys.argv[1]   
    k, p = 20, 0.8
#     [k, p] = sys.argv[2:4]
    k, p = int(k), float(p)

    import config
    results = main(query, 
                   config.command_embedding_pickle,
                   config.method, 
                   config.model, k, p)
    for (command, score) in results:
        print (command.ljust(40), f"{score:.3f}")
