import sys
import pickle
import subprocess
import numpy as np
import networkx as nx
import pygraphviz
from collections import Counter, defaultdict

if len(sys.argv) == 1:
    path_to_dag = 'dots/1A_UMBC_tokenized.txt_100_sg.vec.gz_True_200_0.3_unit_True_vocabulary_filtered.alph.reduced2_more_permissive.dot'
    print('WARNING: No dot input file provided, thus defaulting to the usage of {}'.format(path_to_dag))
else:
    path_to_dag = sys.argv[1]

dataset_dir = '/home/berend/datasets/semeval2018/SemEval18-Task9'
dataset_id = path_to_dag.replace('dots/', '')[0:2]
is_sg = '_sg' in path_to_dag

dataset_mapping = {
    '1A':['english', 'UMBC'],
    '1B':['italian', 'it_itwac'],
    '1C':['spanish', 'es_1Billion'],
    '2A':['medical', 'med_pubmed'],
    '2B':['music', 'music_bioreviews']
}

def generate_file_names():
    data_file = '{}/training/data/{}.{}.training.data.txt'.format(
        dataset_dir,
        dataset_id,
        dataset_mapping[dataset_id][0]
    )
    gold_file = '{}/training/gold/{}.{}.training.gold.txt'.format(
        dataset_dir,
        dataset_id,
        dataset_mapping[dataset_id][0]
    )
    vocab_file = '{}/vocabulary/{}.{}.vocabulary.txt'.format(
        dataset_dir,
        dataset_id,
        dataset_mapping[dataset_id][0]
    )
    frequency_file = '{}/SemEval2018_Frequency_lists/{}_{}_frequencylist.txt'.format(
        dataset_dir,
        dataset_id,
        dataset_mapping[dataset_id][0]
    )
    return data_file, gold_file, vocab_file, frequency_file

train_data_file, train_gold_file, vocab, freq_file = generate_file_names()
dev_data_file, dev_gold_file = train_data_file.replace('training', 'trial'), train_gold_file.replace('training', 'trial')
test_data_file = train_data_file.replace('training', 'test')

train_queries = [(l.split('\t')[0].replace(' ', '_'), l.split('\t')[1].strip()) for l in open(train_data_file)]
train_golds = [
    [x.replace(' ', '_') for x in line.strip().split('\t')] for line in open(train_gold_file)
]

gold_counter = defaultdict(Counter)
for tq, tgs in zip(train_queries, train_golds):
    gold_counter[tq[1]].update(tgs)

dev_queries = [(l.split('\t')[0].replace(' ', '_'), l.split('\t')[1].strip()) for l in open(dev_data_file)]
dev_golds = [
    [x.replace(' ', '_') for x in line.strip().split('\t')] for line in open(dev_gold_file)
]

test_queries = [(l.split('\t')[0].replace(' ', '_'), l.split('\t')[1].strip()) for l in open(test_data_file)]

# read in useful data re the data
i2w = {i:w.strip() for i,w in enumerate(open('data/{}.vocab'.format(dataset_id)))}
w2i = {v:k for k,v in i2w.items()}
word_frequencies = {}
for l in open(freq_file):
    word = l.split('\t')[0].replace(' ', '_')
    freq = int(l.split('\t')[1])
    if word.lower() not in word_frequencies:
        word_frequencies[word.lower()] = freq
    word_frequencies[word] = freq
embedding_file = 'dense_embeddings/{}_sg_vocab_filtered.emb'.format(dataset_id, 'sg' if is_sg else 'cbow', 'rb')
embeddings = pickle.load(open(embedding_file, 'rb'))
unit_embeddings = embeddings.copy()
model_row_norms = np.sqrt((unit_embeddings**2).sum(axis=1))[:, np.newaxis]
unit_embeddings /= model_row_norms

dag = nx.drawing.nx_agraph.read_dot(path_to_dag)

deepest_occurrence = {}    # dict mapping words to their location according to their most specific concept 
nodes_to_attributes = {}   # dict containing which neurons are active for a given node
nodes_to_words = {}        # dict containing all the words located at a given node
words_to_nodes = defaultdict(set)        # a dict containing all the nodes a word is assigned to
words_to_attributes = {}   # dict containing the full set of basis active for a given word
for n in dag.nodes(data=True):
    words = n[1]['label'].split('|')[1].split('\\n')
    node_id = int(n[1]['label'].split('|')[0])
    attributes = [att for att in n[1]['label'].split('|')[2].split('\\n') if len(att.strip())>0]
    nodes_to_attributes[node_id] = attributes
    nodes_to_words[node_id] = set(words)

    for w in words:
        words_to_nodes[w].add(node_id)
        if w not in deepest_occurrence or deepest_occurrence[w][2] < len(attributes):
            deepest_occurrence[w] = (node_id, len(words), len(attributes))
            words_to_attributes[w] = attributes

def get_children_words(graph, node_id):
    return [nodes_to_words[int(n.replace('node', ''))] for n in graph['node{}'.format(node_id)].keys()]

def get_own_words(graph, node_id):
    own_words = nodes_to_words[node_id].copy()
    to_remove = set()
    for c in get_children_words(graph, node_id):
        to_remove |= c
    own_words -= to_remove
    return own_words

def update_dag_based_features(features, query_type, gold, own_query_words):
    if gold in own_query_words:
        features['dag_shortest_path'][query_type].append(0)
        features['dag_avg_path_len'][query_type].append(0)
        features['dag_number_of_paths'][query_type].append(1)
    else:
        gold_location = deepest_occurrence[gold][0] if gold_in_dag else 0
        all_paths = list(nx.all_simple_paths(dag, 'node{}'.format(gold_location), 'node{}'.format(query_location)))
        if len(all_paths) > 0:
            features['dag_shortest_path'][query_type].append(min([len(p)-1 for p in all_paths]))
            features['dag_avg_path_len'][query_type].append(np.mean([len(p)-1 for p in all_paths]))
            features['dag_number_of_paths'][query_type].append(len(all_paths))
        else:
            all_paths = list(nx.all_simple_paths(dag, 'node{}'.format(query_location), 'node{}'.format(gold_location)))
            if len(all_paths) == 0:
                features['dag_shortest_path'][query_type].append(-100)
                features['dag_avg_path_len'][query_type].append(-100)
                features['dag_number_of_paths'][query_type].append(0)
            else:
                features['dag_shortest_path'][query_type].append(-min([len(p)-1 for p in all_paths]))
                features['dag_avg_path_len'][query_type].append(-np.mean([len(p)-1 for p in all_paths]))
                features['dag_number_of_paths'][query_type].append(len(all_paths))

def generate_candidates(word):
    w = embeddings[w2i[word]]
    uw = unit_embeddings[w2i[word]]

features = {
            'difference_length' : defaultdict(list),
            'right_above_in_dag' : defaultdict(list),
            'right_below_in_dag' : defaultdict(list),
            'same_dag_position' : defaultdict(list),
            'has_textual_overlap' : defaultdict(list),
            'freq_ratios_log' : defaultdict(list),
            'length_ratios' : defaultdict(list),
            'attribute_differenceA' : defaultdict(list),
            'attribute_differenceB' : defaultdict(list),
            'cosines' : defaultdict(list),
#            'dag_shortest_path' : defaultdict(list),
#            'dag_number_of_paths' : defaultdict(list),
#            'dag_avg_path_len' : defaultdict(list)
}
positive_training_pairs = defaultdict(list)

missed_query, missed_hypernyms = 0, 0
for i, query_tuple, hypernyms in zip(range(len(train_queries)), train_queries, train_golds):
    query, query_type = query_tuple[0], query_tuple[1]
    if query not in w2i:
        missed_query += 1
        missed_hypernyms += len(hypernyms)
        continue
    query_tokens = set(query.lower().split('_'))
    query_in_dag = query in deepest_occurrence
    query_location = deepest_occurrence[query][0] if query_in_dag else 0
    if query_in_dag:
        own_query_words = get_own_words(dag, query_location)
    else: # if the query is not in the dag, it means that it had no nonzero coefficient in its representation
        own_query_words = set(w2i.keys()) - deepest_occurrence.keys()
    for gold in hypernyms:
        if gold not in w2i:
            missed_hypernyms += 1
            continue

        gold_tokens = set(gold.lower().split('_'))
        features['has_textual_overlap'][query_type].append(1 if len(gold_tokens & query_tokens) > 0 else 0)

        gold_in_dag = gold in deepest_occurrence
        gold_location = deepest_occurrence[gold][0] if gold_in_dag else 0
        #update_dag_based_features(features, query_type, gold, own_query_words)
        features['same_dag_position'][query_type].append(1 if query_location == gold_location else 0)
        features['right_below_in_dag'][query_type].append(1 if dag.has_edge('node{}'.format(query_location), 'node{}'.format(gold_location)) else 0)
        features['right_above_in_dag'][query_type].append(1 if dag.has_edge('node{}'.format(gold_location), 'node{}'.format(query_location)) else 0)

        query_vec = embeddings[w2i[query]]
        query_attributes = set(words_to_attributes[query] if query_in_dag else [])
        gold_vec = embeddings[w2i[gold]]
        gold_attributes = set(words_to_attributes[gold] if gold_in_dag else [])
        positive_training_pairs[query_type].append((query, gold))
        features['difference_length'][query_type].append(np.linalg.norm(query_vec - gold_vec))
        features['length_ratios'][query_type].append(np.linalg.norm(query_vec) / np.linalg.norm(gold_vec))
        features['cosines'][query_type].append(unit_embeddings[w2i[query]] @ unit_embeddings[w2i[gold]])
        attribute_intersection_size = len(query_attributes & gold_attributes)
        attribute_union_size = len(query_attributes | gold_attributes)
        features['attribute_differenceA'][query_type].append(len(query_attributes - gold_attributes))
        features['attribute_differenceB'][query_type].append(len(gold_attributes - query_attributes))
        if query in word_frequencies and gold in word_frequencies:
            features['freq_ratios_log'][query_type].append(np.log10(word_frequencies[query] / word_frequencies[gold]))
        else:
            print(query, gold, query in word_frequencies, query in word_frequencies and gold in word_frequencies)


for f in sorted(features.keys()):
    print(f)
    for category in ['Concept', 'Entity']:
        plt.hist(features[f][category])
    plt.legend(['Concept', 'Entity'])
    plt.show()


### provide a baseline ###
out_file = open('{}_baseline.predictions'.format(dataset_id), 'w')
for query_tuple, hypernyms in zip(train_queries, train_golds):
    out_file.write('{}\n'.format('\t'.join([t[0] for t in gold_counter[query_tuple[1]].most_common(15)])))
out_file.close()
subprocess.call(['python2', 'official-scorer.py', '../SemEval18-Task9/training/gold/1A.english.training.gold.txt', '{}_baseline.predictions'.format(dataset_id)])

'''
for query_tuple, hypernyms in zip(train_queries, train_golds):
    query, query_type = query_tuple[0], query_tuple[1]
    if query in deepest_occurrence:
        alternatives_by_distance = {}
        query_location = deepest_occurrence[query][0]
        all_words_of_query_to_filter = get_own_words(dag, query_location)
        all_words_of_query_to_filter.remove(query)
        alternatives_by_distance[0] = all_words_of_query_to_filter

        alternatives_by_distance[1] = set()
        for parent_node in dag_reversed['node{}'.format(query_location)].keys():
            alternatives_by_distance[1].update(get_own_words(dag, int(parent_node.replace('node', ''))))

        alternatives_by_distance[-1] = set()
        for child_node in dag['node{}'.format(query_location)].keys():
            alternatives_by_distance[-1].update(get_own_words(dag, int(child_node.replace('node', ''))))

        for d in [-1, 0, 1]:
            print(query, d, len(alternatives_by_distance[d]), len(alternatives_by_distance[d] & set(hypernyms)))

        for gold in hypernyms:
            if not gold in deepest_occurrence:
                continue
            hypernym_pairs_included.append((query, gold))
            gold_location = deepest_occurrence[gold][0]
            if gold_location == query_location:
                shortest_path_lengths.append(0)
                paths.append([query_location])
                out_file2.write(gold.replace('_', ' ')+'\t')
            else:
                all_paths = list(nx.all_simple_paths(dag_reversed, 'node{}'.format(query_location), 'node{}'.format(gold_location)))
                paths.append(all_paths)
                if len(all_paths) == 0:
                    all_paths_down = list(nx.all_simple_paths(dag, 'node{}'.format(query_location), 'node{}'.format(gold_location)))
                    if len(all_paths_down) == 0:
                        print(query, gold, words_to_attributes[query], words_to_attributes[gold])
                        chance = False
                        shortest_path_lengths.append(-9999)
                    else:
                        shortest_path_lengths.append(-min([len(p)-1 for p in all_paths_down]))
                else:
                    shortest_path_lengths.append(min([len(p)-1 for p in all_paths]))




shortest_path_lengths = []
hypernym_pairs_included = []
paths = []
chances = []
out_file = open('{}.predictions'.format(path_to_dag), 'w')
out_file2 = open('{}_baseline3.predictions'.format(dataset_id), 'w')
alert_counter, missing_word_counter = 0, 0
numbers_predicted = []
numbers_chosen_from = []
cntr = 0
for query_tuple, hypernyms in zip(train_queries, train_golds):
    cntr+=1
    print(cntr)
    predicted_words = 0
    query, query_type = query_tuple[0], query_tuple[1]
    if query in deepest_occurrence:
        query_location = deepest_occurrence[query][0]
        chance = True # we start out being optimistic
        all_words_of_query_to_filter = get_own_words(dag, query_location)
        all_words_of_query_to_filter.remove(query)
        chosen_from = len(all_words_of_query_to_filter)

        if len(all_words_of_query_to_filter) == 0:
            alert_counter += 1
            #print('{}\t{}'.format(alert_counter, query))
        for guess in all_words_of_query_to_filter:
            if predicted_words < 15:
                out_file.write(guess.replace('_', ' ') + '\t')
                predicted_words += 1
                predicted_words += 1
            else:
                break

        parent_nodes = dag_reversed['node{}'.format(query_location)].keys()
        potential_hypernyms = set()
        for parent_node in parent_nodes:
            potential_hypernyms |= get_own_words(dag, int(parent_node.replace('node', '')))

        chosen_from += len(potential_hypernyms)
        for guess in potential_hypernyms:
            if predicted_words < 15:
                predicted_words += 1
                out_file.write(guess.replace('_', ' ') + '\t')
            else:
                break
        numbers_chosen_from.append(chosen_from)

        for gold in hypernyms:
            if not gold in deepest_occurrence:
                continue
            hypernym_pairs_included.append((query, gold))
            gold_location = deepest_occurrence[gold][0]
            if gold_location == query_location:
                shortest_path_lengths.append(0)
                paths.append([query_location])
                out_file2.write(gold.replace('_', ' ')+'\t')
            else:
                all_paths = list(nx.all_simple_paths(dag_reversed, 'node{}'.format(query_location), 'node{}'.format(gold_location)))
                paths.append(all_paths)
                if len(all_paths) == 0:
                    all_paths_down = list(nx.all_simple_paths(dag, 'node{}'.format(query_location), 'node{}'.format(gold_location)))
                    if len(all_paths_down) == 0:
                        print(words_to_attributes[query], words_to_attributes[gold])
                        chance = False
                        shortest_path_lengths.append(-9999)
                    else:
                        shortest_path_lengths.append(-min([len(p)-1 for p in all_paths_down]))
                        if shortest_path_lengths[-1] == -1:
                            out_file2.write(gold.replace('_', ' ')+'\t')
                else:
                    shortest_path_lengths.append(min([len(p)-1 for p in all_paths]))
                    if shortest_path_lengths[-1] == 1:
                        out_file2.write(gold.replace('_', ' ')+'\t')
        chances.append(1 if chance else 0)
    else:
        numbers_chosen_from.append(0)
        missing_word_counter += 1
        #print("MISSING word #{}: {}".format(missing_word_counter, query))
        is_multiword = '_' in query
        
        if is_multiword and 'University' in query:
            if dataset_id == '1A':
                out_file.write('university')
                out_file2.write('university')
            elif dataset_id == '1B':
                out_file.write('università')
            elif dataset_id == '1C':
                out_file.write('universidad')
        elif is_multiword and query[0].isupper():
            out_file.write('person{}'.format('a' if dataset_id in ['1B', '1C'] else ''))
            out_file2.write('person{}'.format('a' if dataset_id in ['1B', '1C'] else ''))
        
    numbers_predicted.append(predicted_words)
    out_file.write('\n')
    out_file2.write('\n')
out_file.close()
out_file2.close()

c=Counter([spl for spl in shortest_path_lengths])
total_nonzeros = sum(c.values())
print([(x, '{:.2%}'.format(c[x]/total_nonzeros)) for x in sorted(c)], total_nonzeros, sum(chances), missing_word_counter, alert_counter, np.mean(numbers_chosen_from), np.mean(numbers_predicted))
subprocess.call(['python2', 'official-scorer.py', '../SemEval18-Task9/training/gold/1A.english.training.gold.txt', '{}.predictions'.format(path_to_dag)])
'''
