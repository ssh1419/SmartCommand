data_root = "./"
method = "sbert"
model = './pre_trained_models/all-MiniLM-L6-v2'

# If you change the model above, be sure to change accordingly in run_all.sh as well

import os
model_name_base=os.path.basename(model)
# command_embedding_pickle = os.path.join(data_root, f"command_embeddings_{method}_{model_name_base}.pkl")
command_embedding_pickle = os.path.join("CombinedPickleCommands.pkl")
