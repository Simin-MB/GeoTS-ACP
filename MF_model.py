
"Multi Guassian Model"

import math
import numpy as np

from collections import defaultdict
from scipy.stats import multivariate_normal


def dist(loc1, loc2):
    lat1, long1 = loc1.lat, loc1.lng
    lat2, long2 = loc2.lat, loc2.lng
    if abs(lat1 - lat2) < 1e-6 and abs(long1 - long2) < 1e-6:
        return 0.0
    degrees_to_radians = math.pi/180.0
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos(cos)
    earth_radius = 6371
    return arc * earth_radius


class Location(object):
    def __init__(self, id, lat, lng, freq, center=-1):
        self.id = id
        self.lat = lat
        self.lng = lng
        self.freq = freq
        self.center = center


class Center(object):
    def __init__(self):
        self.locations = []
        self.total_freq = 0
        self.distribution = None
        self.mu = None
        self.cov = None
        self.lat = None
        self.lng = None

    def add(self, loc):
        self.locations.append(loc)
        self.total_freq += loc.freq

    def build_gaussian(self):
        coo_seq = []
        for loc in self.locations:
            for _ in range(int(loc.freq)):
                coo_seq.append(np.array([loc.lat, loc.lng]))
        coo_seq = np.array(coo_seq)
        self.mu = np.mean(coo_seq, axis=0)
        self.cov = np.cov(coo_seq.T)
        self.distribution = multivariate_normal(self.mu, self.cov, allow_singular=True)
        self.lat = self.mu[0]
        self.lng = self.mu[1]

    def pdf(self, x):
        return self.distribution.pdf(np.array([x.lat, x.lng]))


class MultiGaussianModel(object):
    def __init__(self, alpha=0.02, theta=0.02, dmax=15):
        self.alpha = alpha
        self.theta = theta
        self.dmax = dmax
        self.poi_coos = None
        self.center_list = None

    def build_user_check_in_profile(self, sparse_check_in_matrix):
        L = defaultdict(list)
        for (uid, lid), freq in sparse_check_in_matrix.items():
            lat, lng = self.poi_coos[lid]
            L[uid].append(Location(lid, lat, lng, freq))
        return L

    def discover_user_centers(self, Lu):
        center_min_freq = max(sum([loc.freq for loc in Lu]) * self.theta, 2)
        Lu.sort(key=lambda k: k.freq, reverse=True)
        center_list = []
        center_num = 0
        for i in range(len(Lu)):
            if Lu[i].center == -1:
                center_num += 1
                center = Center()
                center.add(Lu[i])
                Lu[i].center = center_num
                for j in range(i+1, len(Lu)):
                    if Lu[j].center == -1 and dist(Lu[i], Lu[j]) <= self.dmax:
                        Lu[j].center = center_num
                        center.add(Lu[j])
                if center.total_freq >= center_min_freq:
                    center_list.append(center)
        return center_list

    def multi_center_discovering(self, sparse_check_in_matrix, poi_coos):
        self.poi_coos = poi_coos
        L = self.build_user_check_in_profile(sparse_check_in_matrix)

        center_list = {}
        for uid in range(len(L)):
            center_list[uid] = self.discover_user_centers(L[uid])
            for cid in range(len(center_list[uid])):
                center_list[uid][cid].build_gaussian()
        self.center_list = center_list

    def predict(self, uid, lid):
        lat, lng = self.poi_coos[lid]
        l = Location(None, lat, lng, None)

        prob = 0.0
        if uid in self.center_list:
            all_center_freq = sum([cid.total_freq**self.alpha for cid in self.center_list[uid]])
            all_center_pdf = sum([cid.pdf(l) for cid in self.center_list[uid]])
            if not all_center_pdf == 0:
                for cu in self.center_list[uid]:
                    prob += (
                        1.0 / (dist(l, cu) + 1) *
                        (cu.total_freq**self.alpha) / all_center_freq *
                        cu.pdf(l) / all_center_pdf)
        return prob

"Poisson Factor Model"

import time
import math
import numpy as np


class PoissonFactorModel(object):
    def __init__(self, K=30, alpha=20.0, beta=0.2):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.U, self.L = None, None

    def save_model(self, path):
        ctime = time.time()
        print("Saving U and L...",)
        np.save(path + "U", self.U)
        np.save(path + "L", self.L)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_model(self, path):
        ctime = time.time()
        print("Loading U and L...",)
        self.U = np.load(path + "U.npy")
        self.L = np.load(path + "L.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def train(self, sparse_check_in_matrix, max_iters=1, learning_rate=1e-4):
        ctime = time.time()
        print("Training PFM...", )

        alpha = self.alpha
        beta = self.beta
        K = self.K

        F = sparse_check_in_matrix
        M, N = sparse_check_in_matrix.shape
        U = 0.5 * np.sqrt(np.random.gamma(alpha, beta, (M, K))) / K
        L = 0.5 * np.sqrt(np.random.gamma(alpha, beta, (N, K))) / K

        F = F.tocoo()
        entry_index = list(zip(F.row, F.col))

        F = F.tocsr()
        F_dok = F.todok()

        tau = 10
        last_loss = float('Inf')
        for iters in range(max_iters):
            F_Y = F_dok.copy()
            for i, j in entry_index:
                F_Y[i, j] = 1.0 * F_dok[i, j] / U[i].dot(L[j]) - 1
            F_Y = F_Y.tocsr()

            learning_rate_k = learning_rate * tau / (tau + iters)
            U += learning_rate_k * (F_Y.dot(L) + (alpha - 1) / U - 1 / beta)
            L += learning_rate_k * ((F_Y.T).dot(U) + (alpha - 1) / L - 1 / beta)

            loss = 0.0
            for i, j in entry_index:
                loss += (F_dok[i, j] - U[i].dot(L[j]))**2

            print('Iteration:', iters,  'loss:', loss)

            if loss > last_loss:
                print("Early termination.")
                break
            last_loss = loss

        print("Done. Elapsed time:", time.time() - ctime, "s")
        self.U, self.L = U, L

    def predict(self, uid, lid, sigmoid=False):
        if sigmoid:
            return 1.0 / (1 + math.exp(-self.U[uid].dot(self.L[lid])))
        return self.U[uid].dot(self.L[lid])

"Time Aware MF"

import time
import numpy as np
import scipy.sparse as sparse


class TimeAwareMF(object):
    def __init__(self, K, Lambda, alpha, beta, T):
        self.K = K
        self.T = T
        self.Lambda = Lambda
        self.alpha = alpha
        self.beta = beta
        self.U = None
        self.L = None
        self.LT = None

    def save_model(self, path):
        ctime = time.time()
        print("Saving U and L...",)
        for i in range(self.T):
            np.save(path + "U" + str(i), self.U[i])
        np.save(path + "L", self.L)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_model(self, path):
        ctime = time.time()
        print("Loading U and L...",)
        self.U = [np.load(path + "U%d.npy" % i) for i in range(self.T)]
        self.L = np.load(path + "L.npy")
        self.LT = self.L.T
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_sigma(self, path):
        ctime = time.time()
        print("Loading sigma...",)
        sigma = np.load(path + "sigma.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")
        return sigma

    def get_t_1(self, t):
        return (t - 1) if not t == 0 else (self.T - 1)

    def get_phi(self, C, i, t):
        t_1 = self.get_t_1(t)
        norm_t = np.linalg.norm(C[t][i, :].toarray(), 'fro')
        norm_t_1 = np.linalg.norm(C[t_1][i, :].toarray(), 'fro')
        if norm_t == 0 or norm_t_1 == 0:
            return 0.0
        return C[t][i, :].dot(C[t_1][i, :].T)[0, 0] / norm_t / norm_t_1

    def init_sigma(self, C, M, T):
        ctime = time.time()
        print("Initializing sigma...",)
        sigma = [np.zeros(M) for _ in range(T)]
        for t in range(T):
            C[t] = C[t].tocsr()
            for i in range(M):
                sigma[t][i] = self.get_phi(C, i, t)
        sigma = [sparse.dia_matrix(sigma_t) for sigma_t in sigma]
        print("Done. Elapsed time:", time.time() - ctime, "s")
        return sigma

    def train(self, sparse_check_in_matrices, max_iters=100, load_sigma=False):
        Lambda = self.Lambda
        alpha = self.alpha
        beta = self.beta
        T = self.T
        K = self.K

        C = sparse_check_in_matrices
        M, N = sparse_check_in_matrices[0].shape

        if load_sigma:
            sigma = self.load_sigma("C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\MF_social_model\\VSC_project_test_jaccard\\sigma")
        else:
            sigma = self.init_sigma(C, M, T)
            np.save("C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\MF_social_model\\tmp\\sigma", sigma)

        U = [np.random.rand(M, K) for _ in range(T)]
        L = np.random.rand(N, K)

        C = [Ct.tocoo() for Ct in C]
        entry_index = [zip(C[t].row, C[t].col) for t in range(T)]

        C_est = [Ct for Ct in C]
        C = [Ct.tocsr() for Ct in C]

        for iters in range(max_iters):
            for t in range(T):
                C_est[t] = C_est[t].todok()
                for i, j in entry_index[t]:
                    C_est[t][i, j] = U[t][i].dot(L[j])
                C_est[t] = C_est[t].tocsr()

            for t in range(T):
                t_1 = self.get_t_1(t)
                numerator = C[t] * L + Lambda * sigma[t] * U[t_1]
                denominator = np.maximum(1e-6, C_est[t] * L + Lambda * sigma[t] * U[t_1] + alpha * U[t_1])
                U[t] *= np.sqrt(1.0 * numerator / denominator)

            numerator = np.sum([C[t].T * U[t] for t in range(T)], axis=0)
            denominator = np.maximum(1e-6, np.sum([C_est[t].T * U[t]], axis=0) + beta * L)
            L *= np.sqrt(1.0 * numerator / denominator)

            error = 0.0
            for t in range(T):
                C_est_dok = C_est[t].todok()
                C_dok = C[t].todok()
                for i, j in entry_index[t]:
                    error += (C_est_dok[i, j] - C_dok[i, j]) * (C_est_dok[i, j] - C_dok[i, j])
            print('Iteration:', iters, error)
        self.U, self.L = U, L
        self.LT = L.T

    def predict(self, i, j):
        return np.sum([self.U[t][i].dot(self.L[j]) for t in range(self.T)])

"Friend Based CF"

import time
import numpy as np
from collections import defaultdict
import pickle


class FriendBasedCF(object):
    def __init__(self, eta=0.5):
        self.eta = eta
        self.social_proximity = defaultdict(list)
        self.check_in_matrix = None

    def compute_friend_sim(self, check_in_matrix, social_matrix):
        ctime = time.time()
        print("Precomputing similarity between friends...", )
        self.check_in_matrix = check_in_matrix

        for uid in range(len(social_matrix)) :
            for fid in range(len(social_matrix [uid])):
                if uid < fid:

                    u_social_neighbors = set(social_matrix [uid])
                    f_social_neighbors = set(social_matrix [fid])
                    jaccard_friend = (1.0 * len(u_social_neighbors.intersection(f_social_neighbors)) /
                                    len(u_social_neighbors.union(f_social_neighbors)))

                    u_check_in_neighbors = set(check_in_matrix[uid, :].nonzero()[0])
                    f_check_in_neighbors = set(check_in_matrix[fid, :].nonzero()[0])

                    if len(u_check_in_neighbors.union(f_check_in_neighbors)) == 0:
                        jaccard_check_in = 0
                    else:
                        jaccard_check_in = (1.0 * len(u_check_in_neighbors.intersection(f_check_in_neighbors)) /
                                        len(u_check_in_neighbors.union(f_check_in_neighbors)))

                    if jaccard_friend > 0 and jaccard_check_in > 0:
                        self.social_proximity[uid].append([fid, jaccard_friend, jaccard_check_in])

        print("Done. Elapsed time:", time.time() - ctime, "s")

    def predict(self, i, j):
        if i in self.social_proximity:
            numerator = np.sum([(self.eta * jf + (1 - self.eta) * jc) * self.check_in_matrix[k, j]
                                for k, jf, jc in self.social_proximity[i]])
            return numerator
        return 0.0


"Metrics"

import numpy as np


def mapk(actual, predicted, k):
    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def precisionk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(predicted)


def recallk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(actual)


def ndcgk(actual, predicted):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i,p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    return dcg / idcg

"Recommendation"

import numpy as np
import time
import scipy.sparse as sparse
from collections import defaultdict
import sys
import os

# from lib.PoissonFactorModel import PoissonFactorModel
# from lib.MultiGaussianModel import MultiGaussianModel
# from lib.TimeAwareMF import TimeAwareMF
# from lib.metrics import precisionk, recallk, ndcgk, mapk

def read_friend_data():
    social_data = open(social_file, 'r').readlines()
    social_matrix = np.zeros((user_num, user_num), dtype=int)
    for eachline in social_data:
        uid, fid = eachline.strip().split()
        uid, fid = int(uid), int(fid)
        if uid < user_num and fid < user_num:
            social_matrix[uid, fid] = 1
            social_matrix[fid, uid] = 1
    return social_matrix

def read_poi_coos():
    poi_coos = {}
    poi_data = open(poi_file, 'r').readlines()
    for eachline in poi_data:
        lid, lat, lng = eachline.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poi_coos[lid] = (lat, lng)
    return poi_coos


def read_training_data_2():
    train_data = open(train_file, 'r').readlines()
    training_matrix = np.zeros((user_num, poi_num))
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        training_matrix[uid, lid] = freq
    return training_matrix


def read_training_data():
    # load train data
    train_data = open(train_file, 'r').readlines()
    sparse_training_matrix = sparse.dok_matrix((user_num, poi_num))
    training_matrix = np.zeros((user_num, poi_num))
    training_tuples = set()
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        sparse_training_matrix[uid, lid] = freq
        training_matrix[uid, lid] = 1.0
        training_tuples.add((uid, lid))


def read_training_data():
    # load train data
    train_data = open(train_file, 'r').readlines()
    sparse_training_matrix = sparse.dok_matrix((user_num, poi_num))
    training_tuples = set()
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        sparse_training_matrix[uid, lid] = freq
        training_tuples.add((uid, lid))

    # load checkins
    # time_list_hour = open("./result/time_hour" + ".txt", 'w')
    check_in_data = open(check_in_file, 'r').readlines()
    training_tuples_with_day = defaultdict(int)
    training_tuples_with_time = defaultdict(int)
    for eachline in check_in_data:
        uid, lid, ctime = eachline.strip().split()
        uid, lid, ctime = int(uid), int(lid), float(ctime)
        if (uid, lid) in training_tuples:
            hour = time.gmtime(ctime).tm_hour
            training_tuples_with_time[(hour, uid, lid)] += 1.0
            if 8 <= hour < 18:
                # working time
                hour = 0
            elif hour >= 18 or hour < 8:
                # leisure time
                hour = 1

            training_tuples_with_day[(hour, uid, lid)] += 1.0

    # Default setting: time is partitioned to 24 hours.
    sparse_training_matrices = [sparse.dok_matrix((user_num, poi_num)) for _ in range(24)]
    for (hour, uid, lid), freq in training_tuples_with_time.items():
        sparse_training_matrices[hour][uid, lid] = 1.0 / (1.0 + 1.0 / freq)

    # Default setting: time is partitioned to WD and WE.
    sparse_training_matrix_WT = sparse.dok_matrix((user_num, poi_num))
    sparse_training_matrix_LT = sparse.dok_matrix((user_num, poi_num))

    for (hour, uid, lid), freq in training_tuples_with_day.items():
        if hour == 0:
            sparse_training_matrix_WT[uid, lid] = freq
        elif hour == 1:
            sparse_training_matrix_LT[uid, lid] = freq

    print ("Data Loader Finished!")
    return sparse_training_matrices, sparse_training_matrix, sparse_training_matrix_WT, sparse_training_matrix_LT, training_tuples


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    print("The loading of Ground Truth Finished.")
    return ground_truth


def main():
    sparse_training_matrices, sparse_training_matrix, sparse_training_matrix_WT, sparse_training_matrix_LT, training_tuples = read_training_data()
    ground_truth = read_ground_truth()
    training_matrix = read_training_data_2()
    poi_coos = read_poi_coos()
    #social_relations = read_friend_data()
    social_matrix = read_friend_data()

    start_time = time.time()

    save_social_proximity = False

    PFM.train(sparse_training_matrix, max_iters=10, learning_rate=1e-4)
    # Multi-Center Weekday
    MGMWT.multi_center_discovering(sparse_training_matrix_WT, poi_coos)
    # Multi-Center Weekend
    MGMLT.multi_center_discovering(sparse_training_matrix_LT, poi_coos)

    TAMF.train(sparse_training_matrices, max_iters=30, load_sigma=False)

    if save_social_proximity:

        S.compute_friend_sim(training_matrix, social_matrix)
        with open('social_proximity.pkl', 'wb') as file:
            pickle.dump(S.social_proximity, file)

    else:
        with open('social_proximity.pkl', 'rb') as file:
            S.social_proximity = pickle.load(file)

 
    elapsed_time = time.time() - start_time
    print("Done. Elapsed time:", elapsed_time, "s")

    execution_time = open("C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\MF_social_model\\result_Dec 10_optimal\\execution_time" + ".txt", 'w')
    execution_time.write(str(elapsed_time))

    rec_list = open("C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\MF_social_model\\result_Dec 10_optimal\\reclist_top_" + str(top_k) + ".txt", 'w')
    result_5 = open("C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\MF_social_model\\result_Dec 10_optimal\\result_top_" + str(5) + ".txt", 'w')
    result_10 = open("C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\MF_social_model\\result_Dec 10_optimal\\result_top_" + str(10) + ".txt", 'w')
    result_15 = open("C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\MF_social_model\\result_Dec 10_optimal\\result_top_" + str(15) + ".txt", 'w')
    result_20 = open("C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\MF_social_model\\result_Dec 10_optimal\\result_top_" + str(20) + ".txt", 'w')

    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    # list for different ks
    precision_5,  recall_5,  nDCG_5,  MAP_5 = 0, 0, 0, 0
    precision_10, recall_10, nDCG_10, MAP_10 = 0, 0, 0, 0
    precision_15, recall_15, nDCG_15, MAP_15 = 0, 0, 0, 0
    precision_20, recall_20, nDCG_20, MAP_20 = 0, 0, 0, 0

    if save_predictions:
    
        #caching prediction matrices for future trials
        PFM_predction_cache = np.zeros((user_num, poi_num))
        MGMWT_predction_cache = np.zeros((user_num, poi_num))
        MGMLT_predction_cache = np.zeros((user_num, poi_num))
        TAMF_predction_cache = np.zeros((user_num, poi_num))
        S_predction_cache = np.zeros((user_num, poi_num))
    else:

            with open('PFM_50_users.npy', 'rb') as f:
                PFM_predction_cache = np.load(f)
                print("max & min for PFM", np.min(PFM_predction_cache), np.max(PFM_predction_cache))
           
            with open('MGMWT_50_users.npy', 'rb') as f:
                MGMWT_predction_cache = np.load(f)
                print("max & min for MGMWT", np.min(MGMWT_predction_cache), np.max(MGMWT_predction_cache))
            
            with open('MGMLT_50_users.npy', 'rb') as f:
                MGMLT_predction_cache = np.load(f)
                print("max & min for MGMLT", np.min(MGMLT_predction_cache), np.max(MGMLT_predction_cache))

            with open('TAMF_50_users.npy', 'rb') as f:
                TAMF_predction_cache = np.load(f)
                print("max & min for TAMF", np.min(TAMF_predction_cache), np.max(TAMF_predction_cache))

            with open('S_50_users.npy', 'rb') as f:
                S_predction_cache = np.load(f)
                print("max & min for S", np.min(S_predction_cache), np.max(S_predction_cache))


    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
          
           # What is the meaning of the following structure?
            # overall_scores = [PFM.predict(uid, lid) * (MGMWT.predict(uid, lid) + MGMLT.predict(uid, lid))
            #                  * TAMF.predict(uid, lid) * S.predict(uid, lid)
            #                   if (uid, lid) not in training_tuples else -1
            #                   for lid in all_lids]
                             

            if save_predictions:
                for lid in all_lids:
                    if (uid, lid) not in training_tuples:
                        PFM_predction_cache[uid, lid]=PFM.predict(uid, lid)
                        MGMWT_predction_cache[uid, lid]=MGMWT.predict(uid, lid) 
                        MGMLT_predction_cache[uid, lid]=MGMLT.predict(uid, lid) 
                        TAMF_predction_cache[uid, lid]=TAMF.predict(uid, lid) 
                        S_predction_cache[uid, lid]=S.predict(uid, lid) 
                    else:
                        PFM_predction_cache[uid, lid] = -1
                        MGMWT_predction_cache[uid, lid] = -1
                        MGMLT_predction_cache[uid, lid] = -1
                        TAMF_predction_cache[uid, lid] = -1
                        S_predction_cache[uid, lid] = -1

            overall_scores = [PFM_predction_cache[uid, lid] * (MGMWT_predction_cache[uid, lid] + MGMLT_predction_cache[uid, lid])
                            * TAMF_predction_cache[uid, lid] * S_predction_cache[uid, lid]
                             if (uid, lid) not in training_tuples else -1
                             for lid in all_lids]
                     
            overall_scores = np.array(overall_scores)

            predicted = list(reversed(overall_scores.argsort()))[:top_k]
            actual = ground_truth[uid]

            # calculate the average of different k
            precision_5 = precisionk(actual, predicted[:5])
            recall_5 = recallk(actual, predicted[:5])
            nDCG_5 = ndcgk(actual, predicted[:5])
            MAP_5 = mapk(actual, predicted[:5], 5)

            precision_10 = precisionk(actual, predicted[:10])
            recall_10 = recallk(actual, predicted[:10])
            nDCG_10 = ndcgk(actual, predicted[:10])
            MAP_10 = mapk(actual, predicted[:10], 10)

            precision_15 = precisionk(actual, predicted[:15])
            recall_15 = recallk(actual, predicted[:15])
            nDCG_15 = ndcgk(actual, predicted[:15])
            MAP_15 = mapk(actual, predicted[:15], 15)

            precision_20 = precisionk(actual, predicted[:20])
            recall_20 = recallk(actual, predicted[:20])
            nDCG_20 = ndcgk(actual, predicted[:20])
            MAP_20 = mapk(actual, predicted[:20], 20)

            rec_list.write('\t'.join([
                str(cnt),
                str(uid),
                ','.join([str(lid) for lid in predicted])
            ]) + '\n')

            # write the different ks
            result_5.write('\t'.join([str(cnt), str(uid), str(precision_5), str(recall_5), str(nDCG_5), str(MAP_5)]) + '\n')
            result_10.write('\t'.join([str(cnt), str(uid), str(precision_10), str(recall_10), str(nDCG_10), str(MAP_10)]) + '\n')
            result_15.write('\t'.join([str(cnt), str(uid), str(precision_15), str(recall_15), str(nDCG_15), str(MAP_15)]) + '\n')
            result_20.write('\t'.join([str(cnt), str(uid), str(precision_20), str(recall_20), str(nDCG_20), str(MAP_20)]) + '\n')

    print("<< GeoTS is Finished >>") 

    if save_predictions:


        with open('PFM_50_users.npy', 'wb') as f:
            np.save(f, PFM_predction_cache)

        with open('MGMWT_50_users.npy', 'wb') as f:
            np.save(f, MGMWT_predction_cache)

        with open('MGMLT_50_users.npy', 'wb') as f:
            np.save(f, MGMLT_predction_cache)

        with open('TAMF_50_users.npy', 'wb') as f:
            np.save(f, TAMF_predction_cache)

        with open('S_50_users.npy', 'wb') as f:
            np.save(f, S_predction_cache)
    

if __name__ == '__main__':
    # data_dir = "/content/drive/MyDrive/STACP_model/Gowalla_dataset"

    size_file =  "C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\gowalla_dataset_STACP\\Gowalla_data_size_limited_50_users.txt"
    check_in_file = "C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\gowalla_dataset_STACP\\Gowalla_checkins_limited_50_users.txt"
    train_file =  "C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\gowalla_dataset_STACP\\Gowalla_train_limited_50_users.txt"
    tune_file =  "C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\gowalla_dataset_STACP\\Gowalla_tune.txt"
    test_file =  "C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\gowalla_dataset_STACP\\Gowalla_test.txt"
    poi_file =  "C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\gowalla_dataset_STACP\\Gowalla_poi_coos.txt"
    social_file = "C:\\Users\\simin\\Documents\\Thesis\\M.Sc. Thesis\\gowalla_dataset_STACP\\Gowalla_social_relations.txt"

    
    save_predictions = False
    user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num = int(user_num), int(poi_num)


    top_k = 100

    PFM = PoissonFactorModel(K=30, alpha=20.0, beta=0.2)
    MGMWT = MultiGaussianModel(alpha=0.02, theta=0.02, dmax=15)
    MGMLT = MultiGaussianModel(alpha=0.02, theta=0.02, dmax=15)
    TAMF = TimeAwareMF(K=100, Lambda=0.5, beta=2.0, alpha=2.0, T=24)
    S = FriendBasedCF(eta=0.5)

    # LFBCA = LocationFriendshipBookmarkColoringAlgorithm(alpha=0.85, beta=0.7, epsilon=0.001)
    # FCF = FriendBasedCF()
    # SC  = SocialCorrelation()

    main()