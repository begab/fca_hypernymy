from collections import Counter, defaultdict
import logging
import argparse
import networkx as nx
import numpy as np
import os
import pickle

from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from official_scorer import return_official_scores

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s: (%(lineno)s) %(levelname)s %(message)s")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subtask', dest='dataset_id', default='1A')
    parser.add_argument('--dense_archit', default='sg', choices=['sg', 'cbow'])
    parser.add_argument('--sparse_dimensions', default=200)  # TODO Sparta
    parser.add_argument('--sparse_density', default=.3)
    parser.add_argument('--sparse-new',
                        action='store_true', dest='sparse_new')
    parser.add_argument('--not-include-sparse-feats',
                        action='store_false', dest='include_sparse_feats')
    parser.add_argument('--regularization', type=float, nargs=1, default=[1.0])
    parser.add_argument('--make_test_predictions', action='store_true')
    parser.add_argument('--file_struct', choices=['szeged', 'sztaki'],
                        default='szeged')
    return parser.parse_args()


class ThreeHundredSparsians():
    def __init__(self, args):
        self.args = args
        logging.debug(args)
        self.regularization = self.args.regularization[0]

    def main(self):
        self.init_get_task_data()
        self.get_task_data()
        self.categories = ['Concept', 'Entity']
        self.get_train_hyp_freq()
        self.read_background_word_freq()
        self.get_embed()
        self.get_dag()
        self.attr_pair_freq = defaultdict(int)
        self.get_training_pairs()
        self.train()
        self.init_eval()
        self.make_baseline()
        self.test()

    def init_get_task_data(self):
        self.dataset_mapping = {
            '1A': ['english', 'UMBC'],
            '1B': ['italian', 'it_itwac'],
            '1C': ['spanish', 'es_1Billion'],
            '2A': ['medical', 'med_pubmed'],
            '2B': ['music', 'music_bioreviews']
        }
        self.dag_basename = (
            '{}_{}_tokenized.txt_100_{}.vec.gz_True_{}_{}_unit_True_'
            'vocabulary_filtered{}.alph.reduced2_more_permissive.dot'.format(
                self.args.dataset_id,
                self.dataset_mapping[self.args.dataset_id][1],
                self.args.dense_archit, self.args.sparse_dimensions,
                self.args.sparse_density,
                'NEW' if self.args.sparse_new else ''))

        if self.args.file_struct == 'szeged':
            self.task_dir = ''
            self.dataset_dir = '/home/berend/datasets/semeval2018/SemEval18-Task9'
        else:
            self.task_dir = '/mnt/store/friend/proj/SemEval18-hypernym/'
            self.dataset_dir = os.path.join(self.task_dir, 'SemEval18-Task9')

    def get_queries(self, phase):
        file_path_ptrns = [
            '{d}/{p}/data/{id_}.{c}.{p}.data.txt',
            '{d}/{p}/gold/{id_}.{c}.{p}.gold.txt',
            '{d}/vocabulary/{id_}.{c}.vocabulary.txt',
            '{d}/SemEval2018_Frequency_lists/{id_}_{c}_frequencylist.txt',
        ]
        data_filen, gold_filen1, self.vocab_file, self.frequency_file = (
            str_.format(
                 d=self.dataset_dir, id_=self.args.dataset_id,
                 c=self.dataset_mapping[self.args.dataset_id][0], p=phase)
            for str_ in file_path_ptrns)
        queries = [(l.split('\t')[0].replace(' ', '_'),
                    l.split('\t')[1].strip()) for l in open(data_filen)]
        golds = [[x.replace(' ', '_') for x in line.strip().split('\t')] for
                 line in open(gold_filen1)] if phase != 'test' else None
        return queries, golds

    def get_task_data(self):
        self.train_queries, self.train_golds = self.get_queries('training')
        self.dev_queries, self.dev_golds = self.get_queries('trial')
        self.test_queries, _ = self.get_queries('test')

    def get_train_hyp_freq(self):
        self.gold_counter = defaultdict(Counter)
        for tq, tgs in zip(self.train_queries, self.train_golds):
            self.gold_counter[tq[1]].update(tgs)
        # very_frequent_hypernyms = {
        #     category: set([
        #         h for h, f in self.gold_counter[category].most_common(10)])
        #     for category in self.categories}
        self.frequent_hypernyms = {
            category: set([
                h for h, f in self.gold_counter[category].most_common()])
            for category in self.categories}

    def read_background_word_freq(self):
        logging.info('Reading background word freq...')
        self.word_frequencies = {}
        for i, l in enumerate(open(self.frequency_file)):
            word = l.split('\t')[0].replace(' ', '_')
            freq = int(l.split('\t')[1])
            if word.lower() not in self.word_frequencies:
                self.word_frequencies[word.lower()] = freq
            self.word_frequencies[word] = freq
        if self.args.dataset_id == '1B':
            """
            quick fix to overcome the fact that the freqiecy file and the
            training data contains this word with different capitalization
            """
            self.word_frequencies['equazione_di_Bernoulli'] = 13

    def get_embed(self):
        i2w = {i: w.strip() for i, w in enumerate(open('data/{}.vocab'.format(
            self.args.dataset_id)))}
        self.w2i = {v: k for k, v in i2w.items()}
        embedding_file = os.path.join(
            self.task_dir, 'dense_embeddings',
            '{}_{}_vocab_filtered.emb'.format(
                self.args.dataset_id, self.args.dense_archit))
        self.embeddings = pickle.load(open(embedding_file, 'rb'))
        self.unit_embeddings = self.embeddings.copy()
        model_row_norms = np.sqrt((self.unit_embeddings**2).sum(
            axis=1))[:, np.newaxis]
        self.unit_embeddings /= model_row_norms

    def get_dag(self):
        root, ext = os.path.splitext(self.dag_basename)
        gpickle_dir = os.path.join(self.task_dir, 'gpickle')
        if not os.path.exists(gpickle_dir):
            os.makedirs(gpickle_dir)
        gpickle_fn = os.path.join(gpickle_dir, '{}.gpickle'.format(root))
        if os.path.exists(gpickle_fn):
            logging.info('Loading gpickle...')
            self.dag = nx.read_gpickle(gpickle_fn)
        else:
            logging.info('Reading dot file...')
            self.dag = nx.drawing.nx_agraph.read_dot(
                os.path.join(self.task_dir, 'dots', self.dag_basename))
            nx.write_gpickle(self.dag, gpickle_fn)
        logging.info('Populating dag dicts...')
        self.deepest_occurrence = defaultdict(lambda: [0])
        # deepest_occurrence = {w: most specific location}
        nodes_to_attributes = {}   # {node: active neurons}
        nodes_to_words = {}  # {node: all the words located at it}, not used
        words_to_nodes = defaultdict(set)   # {w: the nodes it is assigned to}
        self.words_to_attributes = {}  # {word: the set of bases active for it}
        for i, n in enumerate(self.dag.nodes(data=True)):
            words = n[1]['label'].split('|')[1].split('\\n')
            if not i % 100000:
                logging.info((i, words))
            node_id = int(n[1]['label'].split('|')[0])
            attributes = [
                int(att.replace('n', ''))
                for att in n[1]['label'].split('|')[2].split('\\n')
                if len(att.strip()) > 0]
            nodes_to_attributes[node_id] = attributes
            nodes_to_words[node_id] = set(words)  # not used

            for w in words:
                words_to_nodes[w].add(node_id)
                if (w not in self.deepest_occurrence
                        or self.deepest_occurrence[w][2] < len(attributes)):
                    self.deepest_occurrence[w] = (node_id, len(words),
                                                  len(attributes))
                    self.words_to_attributes[w] = attributes

    def calculate_features(self, query_word, gold_candidate, query_type,
                           count_att_pairs=False):
        query_vec = self.embeddings[self.w2i[query_word]]
        query_tokens_l = query_word.lower().split('_')
        query_tokens = set(query_tokens_l)
        query_in_dag = query_word in self.deepest_occurrence
        query_location = self.deepest_occurrence[query_word][0] if query_in_dag else 0
        query_attributes = set(self.words_to_attributes[query_word]
                               if query_in_dag else [])
        # if query_in_dag:
        #    own_query_words = get_own_words(self.dag, query_location)
        # else:
        #    # In this case, the query had no nonzero coefficient
        #    own_query_words = set(self.w2i.keys()) -
        #                      self.deepest_occurrence.keys()

        gold_candidate_vec = self.embeddings[self.w2i[gold_candidate]]
        gold_candidate_tokens_l = gold_candidate.lower().split('_')
        gold_candidate_tokens = set(gold_candidate_tokens_l)
        gold_candidate_in_dag = gold_candidate in self.deepest_occurrence
        gold_candidate_location = self.deepest_occurrence[gold_candidate][0]
        gold_candidate_attributes = set(
            self.words_to_attributes[gold_candidate] if gold_candidate_in_dag
            else [])

        feature_vector = {}
        # We use 79-length lines in most of the files, except for the
        # specification of feature_vector values, where long lines remain.
        feature_vector['basis_combinations'] = []
        for q_att in query_attributes:
            for gc_att in gold_candidate_attributes:
                feature_vector['basis_combinations'].append((q_att, gc_att))

        feature_vector['is_frequent_hypernym'] = int(gold_candidate in self.frequent_hypernyms[query_type])
        feature_vector['has_textual_overlap'] = int(len(gold_candidate_tokens & query_tokens) > 0)

        if count_att_pairs:
            for q_att in query_attributes:
                for c_att in gold_candidate_attributes:
                    self.attr_pair_freq[q_att, c_att] += 1

        for name, ind in [('first', 0), ('last', -1)]:
            feature_vector['cand_is_{}_w'.format(name)] = int(query_tokens_l[ind] == gold_candidate)
            feature_vector['same_{}_w'.format(name)] = int(query_tokens_l[ind] == gold_candidate_tokens_l[ind])

        feature_vector['same_dag_position'] = int(query_location == gold_candidate_location)
        feature_vector['right_below_in_dag'] = int(self.dag.has_edge('node{}'.format(query_location), 'node{}'.format(gold_candidate_location)))
        feature_vector['right_above_in_dag'] = int(self.dag.has_edge('node{}'.format(gold_candidate_location), 'node{}'.format(query_location)))
        feature_vector['difference_length'] = np.linalg.norm(query_vec - gold_candidate_vec)
        feature_vector['length_ratios'] = np.linalg.norm(query_vec) / np.linalg.norm(gold_candidate_vec)
        feature_vector['cosines'] = self.unit_embeddings[self.w2i[query_word]].dot(self.unit_embeddings[self.w2i[gold_candidate]])
        attribute_intersection_size = len(query_attributes & gold_candidate_attributes)
        # attribute_union_size = len(query_attributes | gold_candidate_attributes)
        feature_vector['attribute_differenceA'] = len(query_attributes - gold_candidate_attributes)
        feature_vector['attribute_differenceB'] = len(gold_candidate_attributes - query_attributes)
        feature_vector['attributes_intersect'] = int(attribute_intersection_size > 0)
        if query_word in self.word_frequencies and gold_candidate in self.word_frequencies:
            feature_vector['freq_ratios_log'] = np.log10(self.word_frequencies[query_word] / self.word_frequencies[gold_candidate])
        else:
            feature_vector['freq_ratios_log'] = 0
            # logging.debug((query_word, gold_candidate))
        return feature_vector

    def get_training_pairs(self):
        training_pairs = defaultdict(list)

        np.random.seed(400)
        missed_query, missed_hypernyms = 0, 0
        # TODO the following line may be replaced to
        # for i, (query_tuple, hypernyms) in enumerate(zip(train_queries,
        # train_golds)):
        self.train_feats = defaultdict(lambda: defaultdict(list))
        for i, query_tuple, hypernyms in zip(range(len(self.train_queries)),
                                             self.train_queries,
                                             self.train_golds):
            #  if i % 100 == 0:
            #    logging.info('{} training cases covered.'.format(i))
            query, query_type = query_tuple[0], query_tuple[1]
            if query not in self.w2i:
                missed_query += 1
                missed_hypernyms += len(hypernyms)
                continue

            potential_negative_samples = [
                h for h in self.gold_counter[query_type]
                if h not in hypernyms and h in self.word_frequencies]
            if len(potential_negative_samples) > 0:
                negative_samples = np.random.choice(
                    potential_negative_samples, size=min(
                        50, len(potential_negative_samples)), replace=False)
            else:
                negative_samples = []

            for gold_candidate in set(hypernyms) | set(negative_samples):
                if gold_candidate not in self.w2i:
                    missed_hypernyms += 1
                    continue
                training_pairs[query_type].append((query, gold_candidate))
                self.train_feats['class_label'][query_type].append(
                    gold_candidate in hypernyms)
                for feat_name, feat_val in self.calculate_features(
                        query, gold_candidate, query_type,
                        count_att_pairs=True).items():
                    self.train_feats[feat_name][query_type].append(feat_val)

    def train(self):
        def get_sparse_mx(basis_pairs_per_query):
            sparse_data, sparse_indices, sparse_ptrs = [], [], [0]
            for basis_pairs in basis_pairs_per_query:
                sparse_data.extend(len(basis_pairs) * [1.])
                sparse_indices.extend([
                    basis_pair[0] * self.args.sparse_dimensions + basis_pair[1]
                    for basis_pair in basis_pairs])
                sparse_ptrs.append(len(sparse_indices))
            return csr_matrix(
                (sparse_data, sparse_indices, sparse_ptrs),
                shape=(len(sparse_ptrs)-1, self.args.sparse_dimensions**2))

        X_per_category = {c: [] for c in self.categories}
        y_per_category = {}
        for category in self.categories:
            self.feat_names_used = []
            for feature in sorted(self.train_feats):
                if feature == 'class_label':
                    y_per_category[category] = self.train_feats[feature][category]
                elif feature != 'basis_combinations':
                    self.feat_names_used.append(feature)
                    X_per_category[category].append(
                        self.train_feats[feature][category])

        backup_model = None
        self.models = {
            c: make_pipeline(LogisticRegression(C=self.regularization))
            for c in self.categories}
        for category in self.categories:
            sparse_features = get_sparse_mx(
                self.train_feats['basis_combinations'][category])
            if self.args.include_sparse_feats:
                X = hstack([csr_matrix(np.array(X_per_category[category]).T),
                            sparse_features])
            else:
                X = np.array(X_per_category[category]).T

            if X.shape[0] == 0:
                self.models[category] = None
            else:
                self.models[category].fit(X, y_per_category[category])
                backup_model = self.models[category]  # mármint fallback?
                logging.info((category, '  '.join(
                    '{} {:.2}'.format(fea, coeff) for fea, coeff in sorted(
                        list(zip(self.feat_names_used + [
                            '{}_{}'.format(i, j) for i in
                            range(self.args.sparse_dimensions) for j in
                            range(self.args.sparse_dimensions)],
                                 self.models[category].steps[0][1].coef_[0])),
                        key=lambda p: abs(p[1]), reverse=True)[0:20])))

        for category in self.categories:
            if self.models[category] is None:
                self.models[category] = backup_model

    def make_predictions(self, queries, out_file_name):
        num_of_features = len(self.feat_names_used)
        num_of_features += self.args.sparse_dimensions**2 if self.args.include_sparse_feats else 0
        true_class_index = {
            query_type: [
                i for i, c in enumerate(self.models[query_type].classes_)
                if c][0]
            for query_type in self.categories}

        with open(out_file_name, 'w') as pred_file:
            for i, query_tuple in zip(range(len(queries)), queries):
                # logging.info(query_tuple, hypernyms)
                if i % 250 == 0:
                    logging.debug('{} predictions made'.format(i))
                query, query_type = query_tuple[0], query_tuple[1]
                if query not in self.w2i:
                    for x in self.gold_counter[query_type].most_common(15):
                        pred_file.write(x[0].replace('_', ' ') + '\t')
                    pred_file.write('\n')
                    continue

                possible_hypernyms = []
                sparse_data, sparse_indices, sparse_ptrs = [], [], [0]
                possible_candidates = [
                    h for h in self.gold_counter[query_type]]
                # TODO shall we regard all the vocabulary as a potential
                # hypernym?
                for gold_candidate in possible_candidates:
                    if gold_candidate not in self.w2i:
                        continue
                    possible_hypernyms.append(gold_candidate)
                    feature_vector = self.calculate_features(
                        query, gold_candidate, query_type)
                    for feat_ind, feat_name in enumerate(self.feat_names_used):
                        sparse_data.append(feature_vector[feat_name])
                        sparse_indices.append(feat_ind)

                    if self.args.include_sparse_feats:
                        basis_pairs = feature_vector['basis_combinations']
                        sparse_data.extend(len(basis_pairs) * [1])
                        sparse_indices.extend([
                            len(self.feat_names_used) + basis_pair[0] *
                            self.args.sparse_dimensions + basis_pair[1]
                            for basis_pair in basis_pairs])
                    sparse_ptrs.append(len(sparse_data))
                features_to_rank = csr_matrix(
                    (sparse_data, sparse_indices, sparse_ptrs),
                    shape=(len(possible_hypernyms), num_of_features))
                possible_hypernym_scores = self.models[
                    query_type].predict_proba(
                        features_to_rank)[:, true_class_index[query_type]]
                possible_hypernyms = [(h, s) for h, s in zip(
                    possible_hypernyms, possible_hypernym_scores)]

                sorted_hypernyms = sorted(
                    possible_hypernyms, key=lambda x: x[1])[-15:]
                sorted_hypernyms = sorted(
                    sorted_hypernyms, key=lambda p:
                    self.word_frequencies[p[0]], reverse=True)
                for prediction in sorted_hypernyms:
                    pred_file.write(prediction[0].replace('_', ' ') + '\t')
                    # logging.debug('\t\t',
                    # possible_hypernyms[prediction_index].replace('_', ' '))
                pred_file.write('\n')

    def init_eval(self):
        self.gold_file = os.path.join(
            self.dataset_dir, 'trial/gold',
            '{}.{}.trial.gold.txt'.format(
                self.args.dataset_id,
                self.dataset_mapping[self.args.dataset_id][0]))
        self.metrics = ['MAP', 'MRR', 'P@1', 'P@3', 'P@5', 'P@15']

    def make_baseline(self):
        """
        predicts the most common training hypernyms per query type
        """
        baseline_filen = '{}_baseline.predictions'.format(self.args.dataset_id)
        with open(baseline_filen, mode='w') as out_file:
            for query_tuple, hypernyms in zip(self.dev_queries,
                                              self.dev_golds):
                out_file.write('{}\n'.format('\t'.join([
                    t[0] for t in
                    self.gold_counter[query_tuple[1]].most_common(15)])))

    def write_metrics(self, prediction_filen, metric_filen):
        results = return_official_scores(self.gold_file, prediction_filen)
        for met in self.metrics:
            logging.info('{} {:.3}'.format(met, results[met]))
        with open(metric_filen, mode='w') as metric_file:
            metric_file.write('{}\t{}\t{}\t{}'.format(
                '\t'.join('{:.3}'.format(results[mtk])
                          for mtk in self.metrics),
                self.regularization, self.args.include_sparse_feats,
                self.dag_basename))

    def test(self):
        def get_out_filen(dev_or_test, pred_or_met):
            out_dir = os.path.join(self.task_dir, 'results', dev_or_test,
                                   pred_or_met)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            return '{}.{}_{}_{}.{}.output.txt'.format(
                os.path.join(out_dir, self.args.dataset_id),
                self.dataset_mapping[self.args.dataset_id][0],
                self.dag_basename,
                self.args.include_sparse_feats,
                self.regularization,
            )
        def eval_on(phase, queries):
            pred_file_name = get_out_filen(phase, 'predictions')
            self.make_predictions(self.dev_queries, pred_file_name)
            self.write_metrics(pred_file_name, get_out_filen(phase, 'metrics'))

        eval_on('dev', self.dev_queries)
        if self.args.make_test_predictions:
            eval_on('test', self.test_queries)

    """
    The rest of the class is attic.
    """

    def logg_attribute_pair_hist(self):
        attribute_pair_hist = defaultdict(int)
        for fq in self.attr_pair_freq.values():
            attribute_pair_hist[fq] += 1
            logging.info((
                len(self.attr_pair_freq),
                sorted(attribute_pair_hist.items(), key=lambda item: item[1],
                       reverse=True)))

    def get_children_words(graph, node_id):
        return [nodes_to_words[int(n.replace('node', ''))]
                for n in graph['node{}'.format(node_id)].keys()]

    def get_own_words(self, graph, node_id):
        own_words = nodes_to_words[node_id].copy()
        to_remove = set()
        for c in self.get_children_words(graph, node_id):
            to_remove |= c
        own_words -= to_remove
        return own_words

    def update_dag_based_features(self, features, query_type, gold, own_query_words):
        # TODO további self.dag-ból jövő jellemzőket is kipróbálni
        """
        ez a metódus teljesen halott kód
        még a logreges kísérletek elején használágattam, de a dag-beli
        útkeresés nagyon le lassította volna az egész folyamatot
        ráadásul eleinte még a gold párosokon kipróbálva az látszott, hogy
        nagyon sokszor nem lenne értelmes eredménye ezeknek a jellemzőknek
        (mert nincs út a query és a gold között többnyire)
        a végén a gyorsabb eredmény érdekében csak annyi jellemzőt vettem a
        self.dag-okból, hogy a (query, gold) páros legmélyebb előfordulása
            * ugyannabbban a csúcsban található-e (same_dag_position), vagy
            * közvetlen egy lépésre
                fölfele (right_above_in_dag) vagy lefele (right_below_in_dag)
        """
        if gold in own_query_words:
            features['dag_shortest_path'][query_type].append(0)
            features['dag_avg_path_len'][query_type].append(0)
            features['dag_number_of_paths'][query_type].append(1)
        else:
            if gold_candidate_in_dag:
                gold_location = self.deepest_occurrence[gold][0]
            else:
                # úgy lett kezelve, mintha a 0-ás csúcsban (a gyökérben) lenne,
                # azaz nem tudnánk semmit az ő attribútumairól
                gold_location = 0
            # TODO undefined names 'gold_in_dag', 'query_location'
            all_paths = list(nx.all_simple_paths(
                self.dag, 'node{}'.format(gold_location),
                'node{}'.format(query_location)))
            if len(all_paths) > 0:
                features['dag_shortest_path'][query_type].append(
                    min([len(p)-1 for p in all_paths]))
                features['dag_avg_path_len'][query_type].append(
                    np.mean([len(p)-1 for p in all_paths]))
                features['dag_number_of_paths'][query_type].append(len(all_paths))
            else:
                all_paths = list(nx.all_simple_paths(
                    self.dag, 'node{}'.format(query_location),
                    'node{}'.format(gold_location)))
                if len(all_paths) == 0:
                    features['dag_shortest_path'][query_type].append(-100)
                    features['dag_avg_path_len'][query_type].append(-100)
                    features['dag_number_of_paths'][query_type].append(0)
                else:
                    features['dag_shortest_path'][query_type].append(
                        -min([len(p)-1 for p in all_paths]))
                    features['dag_avg_path_len'][query_type].append(
                        -np.mean([len(p)-1 for p in all_paths]))
                    features['dag_number_of_paths'][query_type].append(
                        len(all_paths))


if __name__ == '__main__':
    sparsians = ThreeHundredSparsians(get_args())
    sparsians.main()
