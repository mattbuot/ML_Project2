"""Redefinitions of Surprise algorithm for the discrete case (all predictions must be integers)"""

from surprise import *


class SVDDiscrete(SVD):
    def estimate(self, u, i):
        return round(SVD.estimate(self, u, i))


class BaselineOnlyDiscrete(BaselineOnly):
    def estimate(self, u, i):
        return round(BaselineOnly.estimate(self, u, i))


class KNNBasicDiscrete(KNNBasic):
    def estimate(self, u, i):
        est, details = KNNBasic.estimate(self, u, i)
        return round(est), details


class KNNWithMeansDiscrete(KNNWithMeans):
    def estimate(self, u, i):
        est, details = KNNWithMeans.estimate(self, u, i)
        return round(est), details


class KNNWithZScoreDiscrete(KNNWithZScore):
    def estimate(self, u, i):
        est, details = KNNWithZScore.estimate(self, u, i)
        return round(est), details


class KNNBaselineDiscrete(KNNBaseline):
    def estimate(self, u, i):
        est, details = KNNBaseline.estimate(self, u, i)
        return round(est), details


class NMFDiscrete(NMF):
    def estimate(self, u, i):
        return round(NMF.estimate(self, u, i))


class SlopeOneDiscrete(SlopeOne):
    def estimate(self, u, i):
        return round(SlopeOne.estimate(self, u, i))


class CoClusteringDiscrete(CoClustering):
    def estimate(self, u, i):
        return round(CoClustering.estimate(self, u, i))