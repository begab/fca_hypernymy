import sys
import subprocess
import numpy as np
import networkx as nx
import pygraphviz
from collections import Counter, defaultdict

if len(sys.argv) == 1:
    path_to_dag = '1A_UMBC_tokenized.txt_100_sg.vec.gz_True_200_0.3_unit_True_vocabulary_filtered.alph.reduced2_more_permissive.dot'
    print('WARNING: No dot input file provided, thus defaulting to the usage of {}'.format(path_to_dag))
else:
    path_to_dag = sys.argv[1]

dag = nx.drawing.nx_agraph.read_dot(path_to_dag)

deepest_occurrence = {}
attributes_for_nodes = {}
words_at_nodes = {}
for n in dag.nodes(data=True):
    words = n[1]['label'].split('|')[1].split('\\n')
    node_id = int(n[1]['label'].split('|')[0])
    attributes_for_nodes[node_id] = n[1]['label'].split('|')[2].split('\\n')
    words_at_nodes[node_id] = set(words)

    for w in words:
        if not w in deepest_occurrence or deepest_occurrence[w][1] > len(words):
            deepest_occurrence[w] = (node_id, len(words))

def generate_file_names(dataset_dir, dataset_id):
    dataset_mapping = {
        '1A':['english', 'UMBC'],
        '1B':['italian', 'it_itwac'],
        '1C':['spanish', 'es_1Billion'],
        '2A':['medical', 'med_pubmed'],
        '2B':['music', 'music_bioreviews']
    }
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

dataset_dir = '/home/berend/datasets/semeval2018/SemEval18-Task9'
dataset_id = path_to_dag[0:2]
train_data_file, train_gold_file, vocab, freq_file = generate_file_names(dataset_dir, dataset_id)
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
freqs = {line.split('\t')[0].replace('_', ' '):int(line.split('\t')[1]) for line in open(freq_file)}

def get_children_words(graph, node_id):
    return [words_at_nodes[int(n.replace('node', ''))] for n in graph['node{}'.format(node_id)].keys()]

def get_own_words(graph, node_id):
    own_words = words_at_nodes[node_id].copy()
    to_remove = set()
    for c in get_children_words(graph, node_id):
        to_remove |= c
    own_words -= to_remove
    return own_words

### provide a baseline ###
out_file = open('{}_basline.predictions'.format(dataset_id), 'w')
for query_tuple, hypernyms in zip(train_queries, train_golds):
    out_file.write('{}\n'.format('\t'.join([t[0] for t in gold_counter[query_tuple[1]].most_common(15)])))
out_file.close()
subprocess.call(['python2', 'official-scorer.py', '../SemEval18-Task9/training/gold/1A.english.training.gold.txt', '{}_basline.predictions'.format(dataset_id)])

dag_reversed = dag.reverse()
shortest_path_lengths = []
hypernym_pairs_included = []
paths = []
chances = []
out_file = open('{}.predictions'.format(path_to_dag), 'w')
alert_counter, missing_word_counter = 0, 0
numbers_predicted = []
numbers_chosen_from = []
for query_tuple, hypernyms in zip(train_queries, train_golds):
    predicted_words = 0
    query, query_type = query_tuple[0], query_tuple[1]
    if query in deepest_occurrence:
        query_location = deepest_occurrence[query][0]
        chance = False
        all_words_of_query = words_at_nodes[query_location]
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
                chance = True
                shortest_path_lengths.append(0)
                paths.append([query_location])
            else:
                all_paths = list(nx.all_simple_paths(dag_reversed, 'node{}'.format(query_location), 'node{}'.format(gold_location)))
                paths.append(all_paths)
                if len(all_paths) == 0:
                    if gold in all_words_of_query:
                        #print(gold, query)
                        shortest_path_lengths.append(-2)
                    else:
                        shortest_path_lengths.append(-1)
                else:
                    chance = True
                    shortest_path_lengths.append(min([len(p)-1 for p in all_paths]))
        chances.append(1 if chance else 0)
    else:
        numbers_chosen_from.append(0)
        missing_word_counter += 1
        #print("MISSING word #{}: {}".format(missing_word_counter, query))
        is_multiword = '_' in query
        '''
        if is_multiword and 'University' in query:
            if dataset_id == '1A':
                out_file.write('university')
            elif dataset_id == '1B':
                out_file.write('università')
            elif dataset_id == '1C':
                out_file.write('universidad')
        elif is_multiword and query[0].isupper():
            out_file.write('person{}'.format('a' if dataset_id in ['1B', '1C'] else ''))
        '''
    numbers_predicted.append(predicted_words)
    out_file.write('\n')
out_file.close()

c=Counter([spl for spl in shortest_path_lengths])
total_nonzeros = sum(c.values())
print([(x, '{:.2%}'.format(c[x]/total_nonzeros)) for x in sorted(c)], total_nonzeros, sum(chances), missing_word_counter, alert_counter, np.mean(numbers_chosen_from), np.mean(numbers_predicted))
subprocess.call(['python2', 'official-scorer.py', '../SemEval18-Task9/training/gold/1A.english.training.gold.txt', '{}.predictions'.format(path_to_dag)])
