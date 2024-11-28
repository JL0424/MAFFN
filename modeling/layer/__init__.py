# encoding: utf-8

from .center_loss import CenterLoss
from .triplet_loss import CrossEntropyLabelSmooth, TripletLoss, WeightedRegularizedTriplet
from .non_local import Non_local
from .gem_pool import GeneralizedMeanPooling, GeneralizedMeanPoolingP
from .attention import ChannelAttention, SpatialAttention
from .Unified_Cross_Entropy_Loss import Unified_Cross_Entropy_Loss
