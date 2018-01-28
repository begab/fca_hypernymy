import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: (%(lineno)s) %(levelname)s %(message)s")
import sys
import pickle
import subprocess
import numpy as np
import networkx as nx
import os
import pygraphviz
#import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn import svm

if len(sys.argv) == 1:
    path_to_dag = 'dots/1A_UMBC_tokenized.txt_100_sg.vec.gz_True_200_0.4_unit_True_vocabulary_filtered.alph.reduced2_more_permissive.dot'
    logging.warning('No dot input file provided, thus defaulting to the usage of {}'.format(path_to_dag))
else:
    path_to_dag = sys.argv[1]

#dataset_dir = '/home/berend/datasets/semeval2018/SemEval18-Task9'
dataset_dir = '/mnt/permanent/Language/English/Data/SemEval/2018/Hypernym/SemEval2018_task9_test'
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

if dataset_id == '1B':  # quick fix to overcome the fact that the freqiecy file and the training data contains this word with different capitalization
    word_frequencies['equazione_di_Bernoulli'] = 13

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

features = {
            'difference_length' : defaultdict(list),
            'is_frequent_hypernym' : defaultdict(list),
            'right_above_in_dag' : defaultdict(list),
            'right_below_in_dag' : defaultdict(list),
            'same_dag_position' : defaultdict(list),
            'has_textual_overlap' : defaultdict(list),
            'freq_ratios_log' : defaultdict(list),
            'length_ratios' : defaultdict(list),
            'attribute_differenceA' : defaultdict(list),
            'attribute_differenceB' : defaultdict(list),
            'attributes_intersect' : defaultdict(list),
            'cosines' : defaultdict(list),
            'class_label' : defaultdict(list),
#            'dag_shortest_path' : defaultdict(list),
#            'dag_number_of_paths' : defaultdict(list),
#            'dag_avg_path_len' : defaultdict(list)
}
for name in ['first', 'last']:
    features['cand_is_{}_w'.format(name)] = defaultdict(list)
    features['same_{}_w'.format(name)] = defaultdict(list)
training_pairs = defaultdict(list)

categories = ['Concept', 'Entity']
very_frequent_hypernyms = {category: set([h for h,f in gold_counter[category].most_common(10)]) for category in categories}
frequent_hypernyms = {category: set([h for h,f in gold_counter[category].most_common(100)]) for category in categories}
np.random.seed(400)
missed_query, missed_hypernyms = 0, 0
for i, query_tuple, hypernyms in zip(range(len(train_queries)), train_queries, train_golds):
    #if i % 100 == 0:
    #    logging.info('{} training cases covered.'.format(i))
    query, query_type = query_tuple[0], query_tuple[1]
    if query not in w2i:
        missed_query += 1
        missed_hypernyms += len(hypernyms)
        continue
    query_tokens = set(query.lower().split('_'))
    query_tokens_l = query.lower().split('_')
    query_in_dag = query in deepest_occurrence
    query_location = deepest_occurrence[query][0] if query_in_dag else 0
    if query_in_dag:
        own_query_words = get_own_words(dag, query_location)
    else: # if the query is not in the dag, it means that it had no nonzero coefficient in its representation
        own_query_words = set(w2i.keys()) - deepest_occurrence.keys()

    negative_samples = []
    if len(gold_counter[query_type].keys()-hypernyms) > 0: # check if we can generate any negative samples
        negative_samples = np.random.choice([h for h in gold_counter[query_type] if h not in hypernyms and h in word_frequencies], size=25, replace=False)
    else:
        logging.warning('No negative samples are generated for word {}'.format(query))

    for gold_candidate in set(hypernyms) | set(negative_samples):
        if gold_candidate not in w2i:
            missed_hypernyms += 1
            continue

        features['class_label'][query_type].append(gold_candidate in hypernyms)
        features['is_frequent_hypernym'][query_type].append(1 if gold_candidate in frequent_hypernyms[query_type] else 0)
        gold_candidate_tokens = set(gold_candidate.lower().split('_'))
        gold_candidate_tokens_l = gold_candidate.lower().split('_')
        features['has_textual_overlap'][query_type].append(1 if len(gold_candidate_tokens & query_tokens) > 0 else 0)
        for name, ind in [('first', 0), ('last', -1)]:
            features['cand_is_{}_w'.format(name)][query_type].append(int(
                query_tokens_l[ind] == gold_candidate))
            features['same_{}_w'.format(name)][query_type].append(int(
                query_tokens_l[ind] == gold_candidate_tokens_l[ind]))

        gold_candidate_in_dag = gold_candidate in deepest_occurrence
        gold_candidate_location = deepest_occurrence[gold_candidate][0] if gold_candidate_in_dag else 0
        #update_dag_based_features(features, query_type, gold, own_query_words)
        features['same_dag_position'][query_type].append(1 if query_location == gold_candidate_location else 0)
        features['right_below_in_dag'][query_type].append(1 if dag.has_edge('node{}'.format(query_location), 'node{}'.format(gold_candidate_location)) else 0)
        features['right_above_in_dag'][query_type].append(1 if dag.has_edge('node{}'.format(gold_candidate_location), 'node{}'.format(query_location)) else 0)

        query_vec = embeddings[w2i[query]]
        query_attributes = set(words_to_attributes[query] if query_in_dag else [])
        gold_candidate_vec = embeddings[w2i[gold_candidate]]
        gold_candidate_attributes = set(words_to_attributes[gold_candidate] if gold_candidate_in_dag else [])
        training_pairs[query_type].append((query, gold_candidate))
        features['difference_length'][query_type].append(np.linalg.norm(query_vec - gold_candidate_vec))
        features['length_ratios'][query_type].append(np.linalg.norm(query_vec) / np.linalg.norm(gold_candidate_vec))
        features['cosines'][query_type].append(
            np.matmul(unit_embeddings[w2i[query]], unit_embeddings[w2i[gold_candidate]]))
        attribute_intersection_size = len(query_attributes & gold_candidate_attributes)
        attribute_union_size = len(query_attributes | gold_candidate_attributes)
        features['attribute_differenceA'][query_type].append(len(query_attributes - gold_candidate_attributes))
        features['attribute_differenceB'][query_type].append(len(gold_candidate_attributes - query_attributes))
        features['attributes_intersect'][query_type].append(attribute_intersection_size > 0)
        if query in word_frequencies and gold_candidate in word_frequencies:
            features['freq_ratios_log'][query_type].append(np.log10(word_frequencies[query] / word_frequencies[gold_candidate]))
        else:
            features['freq_ratios_log'][query_type].append(0)
            #logging.info(query, gold_candidate, query in word_frequencies, query in word_frequencies and gold_candidate in word_frequencies)

'''
for f in sorted(features.keys()):
    logging.info(f)
    for category in categories:
        plt.hist([v for c, v in zip(features['class_label'][category], features[f][category]) if c==True])
    plt.legend(categories)
    plt.show()

for category in categories:
    for f in features.keys():
        plt.hist([v for c, v in zip(features['class_label'][category], features[f][category]) if c==False])
        plt.hist([v for c, v in zip(features['class_label'][category], features[f][category]) if c==True])
        plt.legend(['False', 'True'])
        plt.savefig('{}_{}.png'.format(f, category))
        plt.close()
'''

X_per_category = {c: [] for c in categories}
y_per_category = {}
for category in categories:
    feature_names_used = []
    for feature in sorted(features):
        if feature == 'class_label':
            y_per_category[category] = features[feature][category]
        else:
            feature_names_used.append(feature)
            X_per_category[category].append(features[feature][category])

joint_X = np.array([[cv for category in categories for cv in features[fn][category]] for fn in feature_names_used]).T
joint_y = [cl for category in categories for cl in features['class_label'][category]]
joint_model = LogisticRegression()
joint_model.fit(joint_X, joint_y)

models = {c: make_pipeline(LogisticRegression()) for c in categories}
#models = {c: make_pipeline(StandardScaler(), PolynomialFeatures(2), LogisticRegression()) for c in categories}
#svm.SVC(kernel='linear', C=1, random_state=0)
for category in categories:
    logging.info(category)
    X = np.array(X_per_category[category]).T
    if X.shape[0] == 0:
        models[category] = joint_model
        logging.info('Warning: joint model has to be used for {}\t{}'.format(category, list(zip(feature_names_used, joint_model.coef_[0]))))
    else:
        #X = poly.fit_transform(X)
        models[category].fit(X, y_per_category[category])
        logging.info((
            category, 
            sorted(list(zip(feature_names_used,
                            models[category].steps[0][1].coef_[0])),
                   key=lambda p: abs(p[1]), reverse=True)))

true_class_index = [i for i,c in enumerate(models[query_type].classes_) if c][0]
pred_file = open('{}.predictions'.format(path_to_dag.replace('dots', 'predictions')), 'w')
joint_pred_file = open('{}.jointpredictions'.format(path_to_dag.replace('dots', 'predictions')), 'w')
for i, query_tuple, hypernyms in zip(range(len(dev_queries)), dev_queries, dev_golds):
    #logging.info(query_tuple, hypernyms)
    query, query_type = query_tuple[0], query_tuple[1]
    if query not in w2i:
        for x in gold_counter[query_type].most_common(15):
            pred_file.write(x[0].replace('_', ' ') + '\t')
            joint_pred_file.write(x[0].replace('_', ' ') + '\t')
            #logging.info('\t'+x[0].replace('_', ' '))
        pred_file.write('\n')
        joint_pred_file.write('\n')
        continue
    query_tokens = set(query.lower().split('_'))
    query_tokens_l = query.lower().split('_')
    query_in_dag = query in deepest_occurrence
    query_location = deepest_occurrence[query][0] if query_in_dag else 0
    if query_in_dag:
        own_query_words = get_own_words(dag, query_location)
    else: # if the query is not in the dag, it means that it had no nonzero coefficient in its representation
        own_query_words = set(w2i.keys()) - deepest_occurrence.keys()

    possible_hypernyms = []
    possible_hypernym_feature_vecs = []
    for gold_candidate in [h for h in gold_counter[query_type]]:
        if gold_candidate not in w2i:
            continue

        possible_hypernyms.append(gold_candidate)
        feature_vector = {}
        feature_vector['is_frequent_hypernym'] = 1 if gold_candidate in frequent_hypernyms[query_type] else 0
        gold_candidate_tokens = set(gold_candidate.lower().split('_'))
        gold_candidate_tokens_l = gold_candidate.lower().split('_')
        feature_vector['has_textual_overlap'] = 1 if len(gold_candidate_tokens & query_tokens) > 0 else 0
        for name, ind in [('first', 0), ('last', -1)]:
            feature_vector['cand_is_{}_w'.format(name)] = int(
                query_tokens_l[ind] == gold_candidate)
            feature_vector['same_{}_w'.format(name)] = int(
                query_tokens_l[ind] == gold_candidate_tokens_l[ind])

        gold_candidate_in_dag = gold_candidate in deepest_occurrence
        gold_candidate_location = deepest_occurrence[gold_candidate][0] if gold_candidate_in_dag else 0
        #update_dag_based_features(features, query_type, gold, own_query_words)
        feature_vector['same_dag_position'] = 1 if query_location == gold_candidate_location else 0
        feature_vector['right_below_in_dag'] = 1 if dag.has_edge('node{}'.format(query_location), 'node{}'.format(gold_candidate_location)) else 0
        feature_vector['right_above_in_dag'] = 1 if dag.has_edge('node{}'.format(gold_candidate_location), 'node{}'.format(query_location)) else 0

        query_vec = embeddings[w2i[query]]
        query_attributes = set(words_to_attributes[query] if query_in_dag else [])
        gold_candidate_vec = embeddings[w2i[gold_candidate]]
        gold_candidate_attributes = set(words_to_attributes[gold_candidate] if gold_candidate_in_dag else [])
        training_pairs[query_type].append((query, gold_candidate))
        feature_vector['difference_length'] = np.linalg.norm(query_vec - gold_candidate_vec)
        feature_vector['length_ratios'] = np.linalg.norm(query_vec) / np.linalg.norm(gold_candidate_vec)
        feature_vector['cosines'] = np.matmul(
            unit_embeddings[w2i[query]], unit_embeddings[w2i[gold_candidate]])
        attribute_intersection_size = len(query_attributes & gold_candidate_attributes)
        attribute_union_size = len(query_attributes | gold_candidate_attributes)
        feature_vector['attribute_differenceA'] = len(query_attributes - gold_candidate_attributes)
        feature_vector['attribute_differenceB'] = len(gold_candidate_attributes - query_attributes)
        feature_vector['attributes_intersect'] = attribute_intersection_size > 0
        if query in word_frequencies and gold_candidate in word_frequencies:
            feature_vector['freq_ratios_log'] = np.log10(word_frequencies[query] / word_frequencies[gold_candidate])
        else:
            logging.info((
                query, gold_candidate, query in word_frequencies, 
                query in word_frequencies and gold_candidate in word_frequencies))
        try:
            possible_hypernym_feature_vecs.append(np.array([feature_vector[f] for f in feature_names_used]))
        except:
            logging.info(feature_vector.keys())
            break
    features_to_rank = np.array(possible_hypernym_feature_vecs)
    possible_hypernym_scores = models[query_type].predict_proba(features_to_rank)
    for prediction_index in np.argsort(possible_hypernym_scores[:, true_class_index])[-15:]:
        pred_file.write(possible_hypernyms[prediction_index].replace('_', ' ') + '\t')
        #logging.info('\t\t', possible_hypernyms[prediction_index].replace('_', ' '))
    possible_hypernym_scores = joint_model.predict_proba(features_to_rank)
    for prediction_index in np.argsort(possible_hypernym_scores[:, true_class_index])[-15:]:
        joint_pred_file.write(possible_hypernyms[prediction_index].replace('_', ' ') + '\t')
    pred_file.write('\n')
    joint_pred_file.write('\n')
pred_file.close()
joint_pred_file.close()

### provide a baseline predicting the most common etalon hypernyms per query type always ###
out_file = open('{}_baseline.predictions'.format(dataset_id), 'w')
for query_tuple, hypernyms in zip(dev_queries, dev_golds):
    out_file.write('{}\n'.format('\t'.join([t[0] for t in gold_counter[query_tuple[1]].most_common(15)])))
out_file.close()

solution_file = os.path.join(
    dataset_dir, 'trial/gold',
    '{}.{}.trial.gold.txt'.format(dataset_id, dataset_mapping[dataset_id][0]))
subprocess.call(['python2', 'official-scorer.py', solution_file, pred_file.name])
logging.info("=============")
subprocess.call(['python2', 'official-scorer.py', solution_file, joint_pred_file.name])
logging.info(":::::::::::::")
subprocess.call(['python2', 'official-scorer.py', solution_file, out_file.name])
