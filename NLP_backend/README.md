# Ranking Algorithm
The ranking algorithm using the SBERT method to match a query with the embeddings of the command phrases and the embeddings of the command titles and rank the similarity of the query and the command by $max(d(e_q,e_c),d(e_q,e_t))$, when the similarity measure function as $d$ the embedding of the query as $e_q$ and the embedding of a command as $e_c$ and the embedding of the command's title as $e_t$.

## Input:
* The generated embeddings are saved in separate pickle files:
  * `CombinedPickleCommands.pkl`
 

## Processing:
* Load the pickle files of the embeddings of commands and titles.
* Encode the query using the SBERT method.
* Measure the similarities between the embedding of the query and commands.
* Measure the similarities between the embedding of the query and commands' titles.
* Rank the similarities by the maximum score between the similarity scores.
* Filter the result using a combination of top-k and top-p.
* Return the top commands.


## Output:
It will return top-k (20) commands based on their scores.

### When the query is `git clone`, the result is:
```
git.clone                                0.745
git.push                                 0.712
git                                      0.696
git.commit                               0.677
git.ignore                               0.658
git.sync                                 0.638
git.pushTo                               0.633
git.init                                 0.624
git.pull                                 0.623
git.close                                0.605
git.clean                                0.598
git.commitAll                            0.592
git.fetch                                0.588
git.cloneRecursive                       0.581
git.createTag                            0.574
git.addRemote                            0.571
git.pushTags                             0.568
git.rename                               0.567
git.undoCommit                           0.556
git.commitEmpty                          0.556
```

## Requirements:
To run the script, you need to have the following packages installed:

* pickle
* json
* numpy
* requests
* sentence_transformers
