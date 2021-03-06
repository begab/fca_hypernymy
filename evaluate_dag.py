from collections import Counter, defaultdict
import logging
import argparse
import os
import pickle
import time

import numpy as np
import networkx as nx
from scipy.sparse import csc_matrix, csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from official_scorer import return_official_scores

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s: (%(lineno)s) %(levelname)s %(message)s"
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subtask', dest='subtask', default='1A',
                        choices=['1A', '1B', '1C', '2A', '2B'])
    parser.add_argument('--dense_vec', default='sg', choices=['sg', 'cbow'])
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--sparse_dim', type=int, default=200)
    parser.add_argument('--negative_samples', type=int, default=50)
    parser.add_argument('--sparse_density', type=float, default=0.3)
    parser.add_argument(
        '--sparse-new', action='store_true', dest='sparse_new',
        help='sparse bases extracted from the dense embedding by a heuristic '
        'procedure similar to Gram-Schmidt orthogonalization')
    # a submission-be nem ilyenek kerültek

    sparse_atts_parser = parser.add_mutually_exclusive_group(required=False)
    sparse_atts_parser.add_argument('--not-sparse-feats',
                                    dest='include_sparse_att_pairs',
                                    action='store_false')
    sparse_atts_parser.add_argument('--sparse-feats',
                                    dest='include_sparse_att_pairs',
                                    action='store_true')
    parser.set_defaults(include_sparse_att_pairs=True)

    candidates_parser = parser.add_mutually_exclusive_group(required=False)
    candidates_parser.add_argument('--not-filter-candidates',
                                   dest='filter_candidates',
                                   action='store_false')
    candidates_parser.add_argument('--filter-candidates',
                                   dest='filter_candidates',
                                   action='store_true')
    parser.set_defaults(filter_candidates=True)

    dag_feature_parser = parser.add_mutually_exclusive_group(required=False)
    dag_feature_parser.add_argument('--not-dag-feats',
                                    dest='use_dag_features',
                                    action='store_false')
    dag_feature_parser.add_argument('--dag-feats',
                                    dest='use_dag_features',
                                    action='store_true')
    parser.set_defaults(use_dag_features=True)

    gpickle_parser = parser.add_mutually_exclusive_group(required=False)
    gpickle_parser.add_argument('--not-save-gpickle',
                                dest='save_gpickle',
                                action='store_false')
    gpickle_parser.add_argument('--save-gpickle',
                                dest='save_gpickle',
                                action='store_true')
    parser.set_defaults(save_gpickle=True)

    baseline_parser = parser.add_mutually_exclusive_group(required=False)
    baseline_parser.add_argument('--not-baseline-only',
                                 dest='baseline_only',
                                 action='store_false')
    baseline_parser.add_argument('--baseline-only',
                                 dest='baseline_only',
                                 action='store_true')
    parser.set_defaults(baseline_only=False)

    parser.add_argument('--regularization', type=float, nargs='+', default=[1.0])
    predict_test_parser = parser.add_mutually_exclusive_group(required=False)
    predict_test_parser.add_argument('--not-include-test',
                                     dest='make_test_predictions',
                                     action='store_false')
    predict_test_parser.add_argument('--include-test',
                                     dest='make_test_predictions',
                                     action='store_true')
    parser.set_defaults(make_test_predictions=True)
    parser.add_argument('--file_struct', choices=['szeged', 'sztaki'],
                        default='szeged')
    return parser.parse_args()


class ThreeHundredSparsians(object):
    def __init__(self, args):
        self.times = defaultdict(list)
        self.args = args
        logging.debug(args)

        self.dataset_mapping = {
            '1A': ['english', 'UMBC'],
            '1B': ['italian', 'it_itwac'],
            '1C': ['spanish', 'es_1Billion'],
            '2A': ['medical', 'med_pubmed'],
            '2B': ['music', 'music_bioreviews']
        }

        self.dag_basename, self.task_dir, self.dataset_dir = self.init_paths()
        self.train_queries, self.train_golds, _ = self.get_queries('training')
        self.dev_queries, self.dev_golds, self.dev_gold_file = self.get_queries('trial')
        self.test_queries, self.test_golds, self.test_gold_file = self.get_queries('test')
        self.metrics = ['MAP', 'MRR', 'P@1', 'P@3', 'P@5', 'P@15']
        self.categories = ['Concept', 'Entity']
        self.w2i, self.embeddings, self.alphas = self.get_embed()
        self.words_to_atts = defaultdict(list)
        self.words_to_atts.update({w: self.alphas.getcol(i).indices
                                   for w, i in self.w2i.items()})
        self.unit_embeddings = self.embeddings.copy()
        row_norms = np.sqrt(
            (self.unit_embeddings**2).sum(axis=1))[:, np.newaxis]
        self.unit_embeddings /= row_norms
        self.word_freqs = self.read_background_word_freq()
        self.gold_counter, self.frequent_hypernyms = self.get_train_hyp_freq()
        self.possible_hypernyms = self.init_potential_hypernyms()

        if self.args.filter_candidates:
            self.test_candidates = self.gold_counter
        else:
            self.test_candidates = {c: self.possible_hypernyms
                                    for c in self.categories}
        if self.args.use_dag_features:
            self.dag, self.deepest_occurrence, self.node_to_words = self.get_dag()
        else:
            self.dag = None

    def main(self, regularizations, repeats):
        if self.args.baseline_only:
            self.test(None, -1)  # this corresponds to running the baselines
        else:
            train_data, feature_ids, labels = self.get_training_pairs()
            for _ in range(repeats):
                for c in regularizations:
                    models = self.train(train_data, feature_ids, labels, c)
                    self.test(models, c)


    def init_potential_hypernyms(self):
        vocab_file = '{}/vocabulary/{}.{}.vocabulary.txt'.format(
            self.dataset_dir, self.args.subtask,
            self.dataset_mapping[self.args.subtask][0])
        potential_hypernyms = set()
        for l in open(vocab_file):
            hyp_candidate = l.strip()
            if hyp_candidate in self.w2i and hyp_candidate in self.word_freqs:
                potential_hypernyms.add(hyp_candidate)
        return potential_hypernyms

    def init_paths(self):
        dag_basename = (
            '{}_{}_tokenized.txt_100_{}.vec.gz_True_{}_{}_unit_True_'
            'vocabulary_filtered{}.alph.reduced2_more_permissive.dot'.format(
                self.args.subtask,
                self.dataset_mapping[self.args.subtask][1],
                self.args.dense_vec, self.args.sparse_dim,
                self.args.sparse_density,
                'NEW' if self.args.sparse_new else ''))

        if self.args.file_struct == 'szeged':
            task_dir = ''
            dataset_dir = '/home/berend/datasets/semeval2018/SemEval18-Task9'
        else:
            task_dir = '/mnt/store/friend/proj/SemEval18-hypernym/'
            dataset_dir = os.path.join(task_dir, 'SemEval18-Task9')
        return dag_basename, task_dir, dataset_dir

    def get_queries(self, phase):
        file_path_ptrns = [
            '{d}/{p}/data/{id_}.{c}.{p}.data.txt',
            '{d}/{p}/gold/{id_}.{c}.{p}.gold.txt',
        ]
        data_filen, gold_filen1 = (
            str_.format(
                d=self.dataset_dir, id_=self.args.subtask,
                c=self.dataset_mapping[self.args.subtask][0],
                p=phase)
            for str_ in file_path_ptrns)
        queries = [(l.split('\t')[0], l.split('\t')[1].strip())
                   for l in open(data_filen)]
        golds = None
        if os.path.exists(gold_filen1):
            golds = [[x for x in line.strip().split('\t')]
                     for line in open(gold_filen1)]
        return queries, golds, gold_filen1

    def get_train_hyp_freq(self):
        gold_c = defaultdict(Counter)
        discarded_gold_c = Counter()
        for tq, tgs in zip(self.train_queries, self.train_golds):
            for g in tgs:
                if g not in self.word_freqs or g not in self.w2i:
                    discarded_gold_c[g] += 1
            gold_c[tq[1]].update([g for g in tgs
                                  if g in self.word_freqs and g in self.w2i])
        freq_hypernyms = {c: set([h for h, f in gold_c[c].most_common(50)])
                          for c in self.categories}
        for dg in discarded_gold_c.items():
            logging.debug((dg, ' discarded gold'))
        return gold_c, freq_hypernyms

    def read_background_word_freq(self):
        logging.debug('Reading background word freq...')
        word_frequencies = Counter()
        freq_file = '{}/SemEval2018_Frequency_lists/{}_{}_frequencylist.txt'.\
            format(self.dataset_dir, self.args.subtask,
                   self.dataset_mapping[self.args.subtask][0])
        for i, l in enumerate(open(freq_file)):
            word = l.split('\t')[0]
            freq = int(l.split('\t')[1])
            if i % 2500000 == 0:
                logging.debug('{} frequencies read in'.format(i))
            if word.lower() not in word_frequencies:
                word_frequencies[word.lower()] = freq
            word_frequencies[word] = freq
        if self.args.subtask == '1B':
            """
            quick fix to overcome the fact that the frequency file and the
            training data contains this word with different capitalization
            """
            word_frequencies['equazione di Bernoulli'] = 13
        return word_frequencies

    def get_embed(self):
        i2w = {i: w.strip().replace('_', ' ') for i, w in enumerate(open(
            'data/{}.vocab'.format(self.args.subtask)
        ))}
        w2i = {v: k for k, v in i2w.items()}
        embedding_file = os.path.join(
            self.task_dir, 'dense_embeddings',
            '{}_{}_vocab_filtered.emb'.format(
                self.args.subtask, self.args.dense_vec))
        embeddings = pickle.load(open(embedding_file, 'rb'))
        alpha_basename = self.dag_basename.replace('_more_permissive.dot', '')
        alpha_path = os.path.join(self.task_dir, 'alphas', alpha_basename)
        alphas = pickle.load(open(alpha_path, 'rb'))
        return w2i, embeddings, alphas

    def get_dag(self):
        root, ext = os.path.splitext(self.dag_basename)
        gpickle_dir = os.path.join(self.task_dir, 'gpickle')
        if not os.path.exists(gpickle_dir):
            os.makedirs(gpickle_dir)
        gpickle_fn = os.path.join(gpickle_dir, '{}.gpickle'.format(root))
        if os.path.exists(gpickle_fn):
            logging.info('Loading gpickle...')
            dag = nx.read_gpickle(gpickle_fn)
        else:
            logging.info('Reading dot file...')
            dag = nx.drawing.nx_agraph.read_dot(
                os.path.join(self.task_dir, 'dots', self.dag_basename))
            if self.args.save_gpickle:
                nx.write_gpickle(dag, gpickle_fn)
        logging.info('Populating dag dicts...')
        deepest_occurrence = defaultdict(lambda: [0, 0])
        # deepest_occurrence = {w: most specific location}
        node_to_words = {}  # {node: all the words located at it}
        for i, n in enumerate(dag.nodes(data=True)):
            words = [w.replace('_', ' ')
                     for w in n[1]['label'].split('|')[1].split('\\n')]
            node_id = int(n[1]['label'].split('|')[0])
            attributes = [
                int(att.replace('n', ''))
                for att in n[1]['label'].split('|')[2].split('\\n')
                if len(att.strip()) > 0]
            node_to_words[node_id] = set(words)

            for w in words:
                if (w not in deepest_occurrence
                        or deepest_occurrence[w][2] < len(attributes)):
                    deepest_occurrence[w] = (node_id, len(words),
                                             len(attributes))
        return dag, deepest_occurrence, node_to_words

    def get_info_re_words(self, words):
        """

        :param words: a list of words to obtain information for
        :return:
        """
        word_ids = [self.w2i[word] for word in words]
        vectors = self.embeddings[word_ids, :]
        vector_norms = np.linalg.norm(vectors, axis=1)
        unit_vectors = self.unit_embeddings[word_ids, :]
        coeffs = self.alphas[:, word_ids]
        word_atts = [self.words_to_atts[word] for word in words]
        freqs = np.array([self.word_freqs[word] for word in words])
        return vectors, vector_norms, unit_vectors, word_atts, coeffs, freqs

    def calc_features(self, query_words, query_types, candidates, *cand_info):
        """
        :param query_words:
        :param query_types:
        :param candidates:
        :param cand_info: a list of pre-computed candidate statistics
        (if certain candidates keep reoccurring it can be a good idea
        to pre-calculate their statistics and pass them)
        :return: feature matrix w/ len(query_words)*len(candidates) instances
        """
        t = time.time()
        dense_q, norms_q, unit_q, atts_q, coeffs_q, freq_q = \
            self.get_info_re_words(query_words)
        self.times['read1'].append(time.time() - t)
        t = time.time()
        if len(cand_info) == 6:
            dense_c, norms_c, unit_c, atts_c, coeffs_c, freq_c = cand_info
        else:
            dense_c, norms_c, unit_c, atts_c, coeffs_c, freq_c = \
                self.get_info_re_words(candidates)
        self.times['read2'].append(time.time() - t)

        quantity_q, quantity_c = dense_q.shape[0], dense_c.shape[0]
        f = defaultdict(list)
        f['diff_features'] = [np.linalg.norm(
            dense_c - np.tile(vec, (quantity_c, 1)), axis=1)
            for vec in dense_q]
        f['norm_ratios'] = np.outer(norms_q, 1/norms_c)
        f['cosines'] = unit_q.dot(unit_c.T)
        f['freq_ratios_log'] = np.log10(np.outer(freq_q, 1/freq_c))

        t = time.time()
        for q, qt in zip(query_words, query_types):
            q_tokens_l = q.lower().split()
            q_tokens_s = set(q_tokens_l)
            for c in candidates:
                c_tokens_l = c.lower().split()
                f['textual_overlap'].append(
                    int(len(set(c_tokens_l) & q_tokens_s) > 0))
                f['is_frequent_hypernym'].append(
                    int(c in self.frequent_hypernyms[qt]))
                for name, ind in [('first', 0), ('last', -1)]:
                    f['cand_is_{}_w'.format(name)].append(
                        int(q_tokens_l[ind] == c))
                    f['same_{}_w'.format(name)].append(
                        int(q_tokens_l[ind] == c_tokens_l[ind]))
        self.times['word'].append(time.time() - t)

        atts_overlap = coeffs_q.T * coeffs_c.todense()
        nnz_q, nnz_c = np.diff(coeffs_q.indptr), np.diff(coeffs_c.indptr)

        f['att_diffA'] = np.tile(nnz_q, (quantity_c, 1)).T - atts_overlap
        f['att_diffB'] = np.tile(nnz_c, (quantity_q, 1)) - atts_overlap
        f['att_intersect'] = (atts_overlap > 0).astype(np.int)

        def get_dag_location(word):
            word_location = 0
            if word in self.deepest_occurrence:
                word_location = self.deepest_occurrence[word][0]
            return word_location

        if self.dag is not None:
            t = time.time()
            for q in query_words:
                q_dag_loc = get_dag_location(q)
                for c in candidates:
                    c_dag_loc = get_dag_location(c)
                    f['same_dag_position'].append(int(q_dag_loc == c_dag_loc))
                    f['child_in_dag'].append(int(self.dag.has_edge(
                        'node{}'.format(q_dag_loc), 'node{}'.format(c_dag_loc))))
                    f['parent_in_dag'].append(int(self.dag.has_edge(
                        'node{}'.format(c_dag_loc), 'node{}'.format(q_dag_loc))))
            self.times['fca'].append(time.time() - t)
        f = {k: np.ravel(v) for k, v in f.items()}

        t = time.time()
        data, indices, ptrs = [], [], []
        if self.args.include_sparse_att_pairs:
            data, indices, ptrs = self.generate_att_pairs(atts_q, atts_c)
        self.times['sparse2'].append(time.time() - t)
        return f, data, indices, ptrs

    def generate_att_pairs(self, atts_q, atts_c):
        data, indices, pointers = [], [], []
        sparse_dim = self.args.sparse_dim
        for i, q in enumerate(atts_q):
            for j, c in enumerate(atts_c):
                att_pairs = [qi * sparse_dim + ci for qi in q for ci in c]
                indices.extend(att_pairs)
                data.extend(len(att_pairs) * [1])
                pointers.append(len(indices))
        return data, indices, pointers

    def generate_att_pairs_slow(self, coeffs_q, coeffs_c):
        sd = self.args.sparse_dim
        data, indices, ptrs = [], [], []
        qc, cc = coeffs_q.tocoo(), coeffs_c.tocoo()
        q_row = np.array([c * sd + r for r, c in zip(qc.row, qc.col)])
        c_col = np.array([c * sd + r for r, c in zip(cc.row, cc.col)])
        Q = csc_matrix((qc.data, (q_row, np.zeros(qc.row.shape))),
                       shape=(coeffs_q.shape[1]*sd, 1))
        C = csr_matrix((cc.data, (np.zeros(cc.row.shape), c_col)),
                       shape=(1, coeffs_c.shape[1]*sd))
        outer = Q * C
        for q_i in range(0, coeffs_q.shape[1] * sd, sd):
            for c_i in range(0, coeffs_c.shape[1] * sd, sd):
                oc = outer[q_i:q_i+sd, c_i:c_i+sd].tocoo()
                indices.extend([sd * r + c for r, c in zip(oc.row, oc.col)])
                data.extend(len(indices) * [1])
                ptrs.append(len(indices))
        return data, indices, ptrs

    def get_training_pairs(self):
        np.random.seed(400)
        f = dict()  # variable to store features
        labels = {c: [] for c in self.categories}
        X = {c: defaultdict(list) for c in self.categories}
        indices = {c: [] for c in self.categories}
        data = {c: [] for c in self.categories}
        ptrs = {c: [0] for c in self.categories}
        stats = {c: [0, 0] for c in self.categories}  # [kept, dropped]
        for i, (query, golds) in enumerate(
                zip(self.train_queries, self.train_golds)):
            if i % 250 == 0:
                for k, v in self.times.items():
                    logging.debug((i, k, np.mean(v), np.sum(v)))
                logging.info('{} training instance processed'.format(i))

            q, q_type = query[0], query[1]
            drop = False
            if q not in self.w2i:
                drop = True
                logging.info('Train query "{}" ({}) not in vocab'.format(
                    q, q_type))
            if q not in self.word_freqs:
                drop = True
                logging.info('Train query "{}" not in freq list'.format(q))
            if drop:
                stats[q_type][1] += 1
                continue
            stats[q_type][0] += 1

            potential_negatives = sorted([
                    h for h in self.gold_counter[q_type] if h not in golds])
            #    h for h in self.possible_hypernyms if h not in golds])
            neg = []  # negative samples
            if len(potential_negatives) > 0:
                neg = np.random.choice(potential_negatives, size=min(
                    self.args.negative_samples, len(potential_negatives)),
                                       replace=False)

            pos = [h for h in golds if h in self.w2i and h in self.word_freqs]
            candidates = pos + [ns for ns in neg]
            labels[q_type].extend(len(pos) * [True] + len(neg) * [False])

            f, d, ind, offsets = self.calc_features([q], [q_type], candidates)
            for feat_name, values in f.items():
                X[q_type][feat_name].append(values)
            indices[q_type].extend(ind)
            data[q_type].extend(d)
            start_ptr = ptrs[q_type][-1]
            for o in offsets:
                ptrs[q_type].append(start_ptr + o)

        logging.info('Training coverage stats: {}'.format(stats))
        feat_names = sorted(f.keys())
        features = dict()
        for ci, c in enumerate(self.categories):
            if stats[c][0] == 0:
                continue  # there is no sample for the given category c
            features[c] = np.array([np.concatenate(X[c][feat])
                                    for feat in feat_names]).T
            sd = self.args.sparse_dim
            if self.args.include_sparse_att_pairs:
                if ci == len(self.categories) - 1:
                    feat_names += ['{}_{}'.format(i, j) for i in range(sd)
                                   for j in range(sd)]
                s = csr_matrix((data[c], indices[c], ptrs[c]),
                               shape=(len(ptrs[c])-1, sd**2))
                features[c] = hstack([features[c], s])
        return features, feat_names, labels

    def train(self, features, feature_names, labels, regularization):
        fallback_model = None
        models = {c: make_pipeline(LogisticRegression(solver='lbfgs',
                                                      max_iter=10000,
                                                      C=regularization))
                  for c in self.categories}
        for cat in self.categories:
            if cat not in features or features[cat].shape[0] == 0:
                models[cat] = None
            else:
                logging.info('training for category {} with {} features'.
                             format(cat, features[cat].shape[1]))
                models[cat].fit(features[cat], labels[cat])
                fallback_model = models[cat]
                logging.info((cat, '  '.join(
                    '{} {:.2}'.format(fea, coeff) for fea, coeff in sorted(
                        zip(feature_names, models[cat].steps[0][1].coef_[0]),
                        key=lambda p: abs(p[1]), reverse=True)[0:20])))

        for category in self.categories:
            if models[category] is None:
                models[category] = fallback_model
        return models

    def make_predictions(self, models, queries, out_filename):
        self.times.clear()
        true_class_index = {
            cat: [i for i, c in enumerate(models[cat].classes_) if c][0]
            for cat in self.categories}

        default_answer = {
            c: '\t'.join([x[0] for x in self.gold_counter[c].most_common(15)])
            for c in self.categories}
        candidate_stats = {c: self.get_info_re_words(words)
                           for c, words in self.test_candidates.items()}

        stats = {c: [0, 0] for c in self.categories}  # [kept, dropped]
        with open(out_filename, 'w') as pred_file:
            for qi, query_tuple in enumerate(queries):
                if qi % 100 == 0:
                    for k, v in self.times.items():
                        logging.debug((qi, k, np.mean(v), np.sum(v)))
                    logging.info('{} cases processed'.format(qi))
                query, cat = query_tuple[0], query_tuple[1]

                if query not in self.w2i:
                    logging.info('Test query "{}" ({}) not in vocab'.format(
                        query, cat))
                    pred_file.write('{}\n'.format(default_answer[cat]))
                    stats[cat][1] += 1
                    continue

                stats[cat][0] += 1
                f, d, ind, pt = self.calc_features(
                    [query], [cat], self.test_candidates[cat],
                    candidate_stats[cat])
                features = np.array([f[feat] for feat in sorted(f.keys())]).T
                if self.args.include_sparse_att_pairs:
                    pt.insert(0, 0)
                    s = csr_matrix((d, ind, pt),
                                   shape=(len(pt)-1, self.args.sparse_dim**2))
                    features = hstack([features, s])

                ci = true_class_index[cat]
                candidate_scores = models[cat].predict_proba(features)[:, ci]
                possible_candidates = [(h, s) for h, s in zip(
                    self.test_candidates[cat], candidate_scores)]

                sorted_candidates = sorted(
                    possible_candidates, key=lambda w: w[1])[-15:]
                sorted_candidates = sorted(
                    sorted_candidates, key=lambda p: self.word_freqs[p[0]],
                    reverse=True)
                pred_file.write('{}\n'.format('\t'.join(
                    [s[0] for s in sorted_candidates])))
        logging.info('Test coverage stats: {}'.format(stats))

    def make_baseline(self, phase, queries, golds, upper_bound):
        """
        Predicts the most common training hypernyms per query type
        (if upper_bound is False)
        if upper_bound is True it predicts the gold hypernyms also found in our vocabulary

        :param phase:
        :param queries:
        :param golds:
        :param upper_bound:
        :return:
        """
        baseline_filename = '{}_{}.{}.predictions'.format(
            self.args.subtask, 'upper' if upper_bound else 'baseline', phase)
        with open(baseline_filename, mode='w') as out_file:
            for query_tuple, hypernyms in zip(queries, golds):
                category = query_tuple[1]
                if upper_bound:
                    out_file.write('{}\n'.format(
                        '\t'.join([h for h in hypernyms
                                   if h in self.test_candidates[category]])))
                else:
                    out_file.write('{}\n'.format('\t'.join([
                        t[0] for t in
                        self.gold_counter[category].most_common(15)])))
        return baseline_filename

    def test(self, models, regularization):
        def get_out_file_name(dev_or_test, pred_or_met):
            out_dir = os.path.join(self.task_dir, 'results', dev_or_test,
                                   pred_or_met)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            return '{}.{}_{}_{}_{}_{}_ns{}_{}.output.txt'.format(
                os.path.join(out_dir, self.args.subtask),
                self.dataset_mapping[self.args.subtask][0],
                self.dag_basename,
                self.args.include_sparse_att_pairs,
                regularization,
                self.args.filter_candidates,
                self.args.negative_samples,
                self.args.use_dag_features,
            )

        def write_metrics(gold_file, pred_filename, metric_filename=None):
            results = return_official_scores(gold_file, pred_filename)
            if metric_filename is not None:
                with open(metric_filename, mode='w') as metric_file:
                    metric_file.write('{}\t{}\t{}\t{}'.format(
                        '\t'.join('{:.3}'.format(results[mtk])
                                  for mtk in self.metrics),
                        regularization, self.args.include_sparse_att_pairs,
                        self.args.use_dag_features, self.dag_basename))
            return results

        def eval_on(phase):
            gold_file = self.test_gold_file
            queries, golds = self.test_queries, self.test_golds
            if phase == 'dev':
                gold_file = self.dev_gold_file
                queries, golds = self.dev_queries, self.dev_golds

            if models is not None:
                predict_fn = get_out_file_name(phase, 'predictions')
                self.make_predictions(models, queries, predict_fn)
                results = write_metrics(gold_file, predict_fn,
                                        get_out_file_name(phase, 'metrics'))
                res_str = '\t'.join(['{:.3}'.format(results[m])
                                     for m in self.metrics])
                params = '\t'.join(map(str,
                                       [self.args.include_sparse_att_pairs,
                                        regularization,
                                        self.args.filter_candidates,
                                        self.args.negative_samples,
                                        self.args.use_dag_features]))
                logging.info('{}\t{}\t{}_{}'.format(params, res_str, phase,
                                                    self.dag_basename))
            else:
                for upper in [True, False]:
                    if (upper and golds is not None) or not upper:
                        baseline_fn = self.make_baseline(phase, queries,
                                                         golds, upper)
                        results = write_metrics(gold_file, baseline_fn)
                        res_str = '\t'.join(['{:.3}'.format(results[m])
                                             for m in self.metrics])
                        logging.info('{}\t{}\t{}_{}_{}_baseline'.format(
                            res_str,
                            '{}FULL'.format(
                                'not' if self.args.filter_candidates else ''),
                            self.args.subtask,
                            phase,
                            'upper' if upper else 'freq'))
        eval_on('dev')
        if self.args.make_test_predictions:
            eval_on('test')

    def get_own_words(self, node_id):
        own_words = self.node_to_words[node_id].copy()
        to_remove = set()
        for c in [self.node_to_words[int(n.replace('node', ''))]
                  for n in self.dag['node{}'.format(node_id)].keys()]:
            to_remove |= c
        own_words -= to_remove
        return own_words

    """
    The rest of the class is attic.
    """
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
            gold_candidate_in_dag = gold in self.deepest_occurrence[gold]
            if gold_candidate_in_dag:
                query_location = self.deepest_occurrence[gold][0]
            else:
                # úgy lett kezelve, mintha a 0-ás csúcsban (a gyökérben) lenne,
                # azaz nem tudnánk semmit az ő attribútumairól
                gold_location = 0
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
    sparsians.main(sparsians.args.regularization, sparsians.args.num_runs)
